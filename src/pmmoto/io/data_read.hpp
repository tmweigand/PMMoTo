#ifndef DATA_READ_HPP
#define DATA_READ_HPP

// You vendored fast_float headers under extern/fast_float/include
// Ensure setup.py adds that path to include_dirs.
#include "fast_float.h"
#include <sys/mman.h>
#include <sys/stat.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <zlib.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// SIMD headers for fast scanning
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

// Branch hints
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PREFETCH_READ(addr) __builtin_prefetch(addr, 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch(addr, 1, 3)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define PREFETCH_READ(addr) ((void)0)
#define PREFETCH_WRITE(addr) ((void)0)
#endif

// A requested column: name as it appears in the "ITEM: ATOMS" header and
// whether to parse it as an integer (int64) or a floating point (double).
struct LammpsColumnSpec
{
    std::string name;
    bool is_int = false;
};

// Only the vector matching is_int is populated.
struct LammpsColumn
{
    std::string name;
    bool is_int = false;
    std::vector<int64_t> int_values;
    std::vector<double> double_values;
};

struct LammpsData
{
    std::vector<LammpsColumn> data_columns; // same order as the requested specs
    std::vector<std::vector<double> > domain_data;
    double timestep = 0.0;
};

struct MMapBuffer
{
    const char* ptr = nullptr;
    size_t len = 0;
    void* map = nullptr;

    MMapBuffer() = default;
    MMapBuffer(const MMapBuffer&) = delete;
    MMapBuffer& operator=(const MMapBuffer&) = delete;

    MMapBuffer(MMapBuffer&& other) noexcept : ptr(other.ptr),
                                              len(other.len),
                                              map(other.map)
    {
        other.ptr = nullptr;
        other.len = 0;
        other.map = nullptr;
    }

    MMapBuffer& operator=(MMapBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (map && len) munmap(map, len);
            ptr = other.ptr;
            len = other.len;
            map = other.map;
            other.ptr = nullptr;
            other.len = 0;
            other.map = nullptr;
        }
        return *this;
    }

    ~MMapBuffer()
    {
        if (map && len) munmap(map, len);
    }
};

static inline MMapBuffer
map_file_readonly(const std::string& filename)
{
    MMapBuffer out;
    int fd = open(filename.c_str(), O_RDONLY);
    if (UNLIKELY(fd < 0)) throw std::runtime_error("open failed: " + filename);

    struct stat st
    {
    };
    if (UNLIKELY(fstat(fd, &st) != 0))
    {
        int e = errno;
        close(fd);
        throw std::runtime_error("fstat failed (" + std::to_string(e) +
                                 "): " + filename);
    }

    size_t size = static_cast<size_t>(st.st_size);
    if (size == 0)
    {
        close(fd);
        return out;
    }

    void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    int e = errno;
    close(fd);
    if (UNLIKELY(p == MAP_FAILED))
    {
        throw std::runtime_error("mmap failed (" + std::to_string(e) +
                                 "): " + filename);
    }

    // Advise kernel we'll read sequentially
    madvise(p, size, MADV_SEQUENTIAL);

    out.map = p;
    out.ptr = static_cast<const char*>(p);
    out.len = size;
    return out;
}

static inline size_t
gzip_uncompressed_size_from_footer(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (UNLIKELY(!ifs))
        throw std::runtime_error("Could not open gz file: " + filename);

    auto end = ifs.tellg();
    if (UNLIKELY(end < std::streamoff(4)))
        throw std::runtime_error("Invalid gz (too small): " + filename);

    ifs.seekg(end - std::streamoff(4));
    unsigned char isize[4]{};
    ifs.read(reinterpret_cast<char*>(isize), 4);
    if (UNLIKELY(!ifs))
        throw std::runtime_error("Failed reading gz footer: " + filename);

    size_t n = size_t(isize[0]) | (size_t(isize[1]) << 8) |
               (size_t(isize[2]) << 16) | (size_t(isize[3]) << 24);
    return n;
}

#ifdef HAVE_LIBDEFLATE
#include <libdeflate.h>
#endif

static inline std::string
readGzipFile_fast_zlib(const std::string& filename)
{
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    // Read the compressed file into memory
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (UNLIKELY(!ifs)) throw std::runtime_error("open gz failed: " + filename);
    size_t comp_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::string comp(comp_size, '\0');
    ifs.read(comp.data(), comp_size);
    if (UNLIKELY(!ifs)) throw std::runtime_error("read gz failed: " + filename);
    ifs.close();

    auto t1 = clock::now();

    size_t out_size = gzip_uncompressed_size_from_footer(filename);
    std::string out(std::max<size_t>(out_size, 1), '\0');

#ifdef HAVE_LIBDEFLATE
    // Fast path: libdeflate
    libdeflate_decompressor* dec = libdeflate_alloc_decompressor();
    if (!dec) throw std::runtime_error("libdeflate_alloc_decompressor failed");

    size_t actual = 0;
    libdeflate_result res = libdeflate_gzip_decompress(
        dec, comp.data(), comp.size(), out.data(), out.size(), &actual);

    if (res == LIBDEFLATE_INSUFFICIENT_SPACE)
    {
        out.resize(out.size() * 2 + 65536);
        res = libdeflate_gzip_decompress(
            dec, comp.data(), comp.size(), out.data(), out.size(), &actual);
    }
    libdeflate_free_decompressor(dec);

    if (res != LIBDEFLATE_SUCCESS)
    {
        throw std::runtime_error("libdeflate_gzip_decompress failed: " +
                                 std::to_string(res));
    }
    out.resize(actual);

    auto t2 = clock::now();

    if (std::getenv("PMMOTO_ZLIB_DEBUG"))
    {
        auto ms = [](auto a, auto b)
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(b - a)
                .count();
        };
        const char* libpath = "?";
#if defined(__APPLE__) || defined(__linux__)
        Dl_info info{};
        if (dladdr((void*)&libdeflate_gzip_decompress, &info) && info.dli_fname)
            libpath = info.dli_fname;
#endif
        std::fprintf(stderr,
                     "[pmmoto.io] gz=%s read=%lldms inflate=%lldms "
                     "zlib=libdeflate lib=%s\n",
                     filename.c_str(),
                     (long long)ms(t0, t1),
                     (long long)ms(t1, t2),
                     libpath);
        std::fflush(stderr);
    }
    return out;
#else
    // Fallback: zlib inflate
    // size_t out_size = gzip_uncompressed_size_from_footer(filename);
    // std::string out;
    // out.resize(out_size);

    z_stream strm{};
    strm.next_in = reinterpret_cast<Bytef*>(comp.data());
    strm.avail_in = static_cast<uInt>(comp.size());
    strm.next_out = reinterpret_cast<Bytef*>(out.data());
    strm.avail_out = static_cast<uInt>(out.size());

    int ret = inflateInit2(&strm, 16 + MAX_WBITS);
    if (UNLIKELY(ret != Z_OK))
        throw std::runtime_error("inflateInit2 failed: " + std::to_string(ret));

    ret = inflate(&strm, Z_FINISH);
    if (UNLIKELY(ret != Z_STREAM_END))
    {
        inflateEnd(&strm);
        if (ret == Z_BUF_ERROR || ret == Z_OK)
        {
            size_t bigger = out.size() * 2 + 65536;
            out.resize(bigger);
            strm = z_stream{};
            strm.next_in = reinterpret_cast<Bytef*>(comp.data());
            strm.avail_in = static_cast<uInt>(comp.size());
            strm.next_out = reinterpret_cast<Bytef*>(out.data());
            strm.avail_out = static_cast<uInt>(out.size());
            ret = inflateInit2(&strm, 16 + MAX_WBITS);
            if (UNLIKELY(ret != Z_OK))
                throw std::runtime_error("inflateInit2 retry failed: " +
                                         std::to_string(ret));
            ret = inflate(&strm, Z_FINISH);
        }
    }
    if (UNLIKELY(ret != Z_STREAM_END))
    {
        inflateEnd(&strm);
        throw std::runtime_error("inflate failed: " + std::to_string(ret));
    }

    size_t actual = out.size() - strm.avail_out;
    inflateEnd(&strm);
    out.resize(actual);
    return out;
#endif
}

// SIMD-accelerated newline finder (AVX2 version)
#ifdef __AVX2__
static inline const char*
find_next_newline_simd(const char* p, const char* end)
{
    const char* aligned_end = p + ((end - p) & ~31);
    __m256i newline = _mm256_set1_epi8('\n');

    while (p < aligned_end)
    {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        __m256i cmp = _mm256_cmpeq_epi8(chunk, newline);
        int mask = _mm256_movemask_epi8(cmp);

        if (mask != 0)
        {
            return p + __builtin_ctz(mask);
        }
        p += 32;
    }

    // Scalar fallback for tail
    while (p < end && *p != '\n') ++p;
    return p;
}
#elif defined(__SSE2__)
static inline const char*
find_next_newline_simd(const char* p, const char* end)
{
    const char* aligned_end = p + ((end - p) & ~15);
    __m128i newline = _mm_set1_epi8('\n');

    while (p < aligned_end)
    {
        __m128i chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m128i cmp = _mm_cmpeq_epi8(chunk, newline);
        int mask = _mm_movemask_epi8(cmp);

        if (mask != 0)
        {
            return p + __builtin_ctz(mask);
        }
        p += 16;
    }

    while (p < end && *p != '\n') ++p;
    return p;
}
#else
static inline const char*
find_next_newline_simd(const char* p, const char* end)
{
    while (p < end && *p != '\n') ++p;
    return p;
}
#endif

// Optimized skip_spaces with unrolled loop
static inline const char*
skip_spaces(const char* p, const char* end)
{
    // Unroll 4x for better throughput
    while (LIKELY(p + 3 < end))
    {
        if (LIKELY(static_cast<unsigned char>(p[0]) > ' ')) return p;
        if (p[0] != ' ' && p[0] != '\t' && p[0] != '\r') return p;

        if (LIKELY(static_cast<unsigned char>(p[1]) > ' ')) return p + 1;
        if (p[1] != ' ' && p[1] != '\t' && p[1] != '\r') return p + 1;

        if (LIKELY(static_cast<unsigned char>(p[2]) > ' ')) return p + 2;
        if (p[2] != ' ' && p[2] != '\t' && p[2] != '\r') return p + 2;

        if (LIKELY(static_cast<unsigned char>(p[3]) > ' ')) return p + 3;
        if (p[3] != ' ' && p[3] != '\t' && p[3] != '\r') return p + 3;

        p += 4;
    }

    while (LIKELY(p < end))
    {
        unsigned char c = static_cast<unsigned char>(*p);
        if (LIKELY(c > ' ')) break;
        if (c != ' ' && c != '\t' && c != '\r') break;
        ++p;
    }
    return p;
}

static inline const char*
skip_token(const char* p, const char* end)
{
    while (LIKELY(p < end))
    {
        unsigned char c = static_cast<unsigned char>(*p);
        if (c <= ' ' && (c == ' ' || c == '\t' || c == '\n' || c == '\r'))
            break;
        ++p;
    }
    return p;
}

static inline const char*
parse_uint64_ptr(const char* p, const char* end, uint64_t& out)
{
    auto res = std::from_chars(p, end, out);
    return res.ptr;
}

static inline const char*
parse_int_ptr(const char* p, const char* end, int64_t& out)
{
    auto res = std::from_chars(p, end, out);
    return res.ptr;
}

static inline const char*
parse_double_ptr(const char* p, const char* end, double& out)
{
    auto res = fast_float::from_chars(p, end, out);
    return res.ptr;
}

static inline void
read_line(const char*& p,
          const char* end,
          const char*& line_start,
          const char*& line_end)
{
    line_start = p;
    p = find_next_newline_simd(p, end);
    line_end = p;
    if (LIKELY(p < end) && *p == '\n') ++p;
    if (UNLIKELY(line_end > line_start && *(line_end - 1) == '\r')) --line_end;
}

static inline std::vector<std::string>
split_tokens(const char* p, const char* end)
{
    std::vector<std::string> out;
    while (true)
    {
        p = skip_spaces(p, end);
        if (p >= end) break;
        const char* t = skip_token(p, end);
        out.emplace_back(p, t);
        p = t;
    }
    return out;
}

static inline bool
is_token_end(const char* p, const char* end)
{
    return p >= end || *p == ' ' || *p == '\t' || *p == '\r' || *p == '\n';
}

static inline void
parse_lammps_atoms_buffer(const char* buf,
                          const char* end,
                          LammpsData& data,
                          const std::vector<LammpsColumnSpec>& specs)
{
    const char* p = buf;
    const char* line_s = nullptr;
    const char* line_e = nullptr;

    read_line(p, end, line_s, line_e);
    read_line(p, end, line_s, line_e);
    {
        double ts = 0.0;
        parse_double_ptr(line_s, line_e, ts);
        data.timestep = ts;
    }

    read_line(p, end, line_s, line_e);
    read_line(p, end, line_s, line_e);
    uint64_t natoms_u = 0;
    parse_uint64_ptr(line_s, line_e, natoms_u);
    size_t natoms = static_cast<size_t>(natoms_u);

    data.domain_data.resize(3, std::vector<double>(2));

    read_line(p, end, line_s, line_e);
    for (int i = 0; i < 3; ++i)
    {
        read_line(p, end, line_s, line_e);
        const char* q = line_s;
        double lo = 0.0, hi = 0.0;
        q = parse_double_ptr(q, line_e, lo);
        q = skip_spaces(q, line_e);
        q = parse_double_ptr(q, line_e, hi);
        data.domain_data[i][0] = lo;
        data.domain_data[i][1] = hi;
    }

    // "ITEM: ATOMS <col> <col> ..." names the columns of every atom line
    read_line(p, end, line_s, line_e);
    static const char atoms_prefix[] = "ITEM: ATOMS";
    const size_t prefix_len = sizeof(atoms_prefix) - 1;
    if (UNLIKELY(static_cast<size_t>(line_e - line_s) < prefix_len ||
                 std::memcmp(line_s, atoms_prefix, prefix_len) != 0))
    {
        throw std::invalid_argument(
            "Expected 'ITEM: ATOMS' header line in LAMMPS file");
    }
    const std::vector<std::string> header =
        split_tokens(line_s + prefix_len, line_e);

    // Map file column index -> requested column index (-1 if not requested)
    std::vector<int> requested_of_col(header.size(), -1);
    size_t last_col = 0;
    for (size_t r = 0; r < specs.size(); ++r)
    {
        auto it = std::find(header.begin(), header.end(), specs[r].name);
        if (UNLIKELY(it == header.end()))
        {
            throw std::invalid_argument("Column '" + specs[r].name +
                                        "' not found in LAMMPS file header");
        }
        size_t col = static_cast<size_t>(it - header.begin());
        requested_of_col[col] = static_cast<int>(r);
        last_col = std::max(last_col, col);
    }

    data.data_columns.resize(specs.size());
    std::vector<int64_t*> int_ptrs(specs.size(), nullptr);
    std::vector<double*> double_ptrs(specs.size(), nullptr);
    for (size_t r = 0; r < specs.size(); ++r)
    {
        LammpsColumn& col = data.data_columns[r];
        col.name = specs[r].name;
        col.is_int = specs[r].is_int;
        if (col.is_int)
        {
            col.int_values.resize(natoms);
            int_ptrs[r] = col.int_values.data();
        }
        else
        {
            col.double_values.resize(natoms);
            double_ptrs[r] = col.double_values.data();
        }
    }

    for (size_t i = 0; i < natoms; ++i)
    {
        if (UNLIKELY(p >= end))
        {
            throw std::invalid_argument(
                "LAMMPS file declares " + std::to_string(natoms) +
                " atoms but ends after " + std::to_string(i));
        }
        read_line(p, end, line_s, line_e);
        PREFETCH_READ(p + 64);

        const char* q = line_s;
        for (size_t c = 0; c <= last_col; ++c)
        {
            q = skip_spaces(q, line_e);
            if (UNLIKELY(q >= line_e))
            {
                throw std::invalid_argument(
                    "Atom line " + std::to_string(i) +
                    " has fewer columns than the file header");
            }

            const int r = requested_of_col[c];
            if (r < 0)
            {
                q = skip_token(q, line_e);
                continue;
            }

            const char* next;
            if (specs[r].is_int)
                next = parse_int_ptr(q, line_e, int_ptrs[r][i]);
            else
                next = parse_double_ptr(q, line_e, double_ptrs[r][i]);

            if (UNLIKELY(next == q || !is_token_end(next, line_e)))
            {
                throw std::invalid_argument(
                    "Could not parse column '" + specs[r].name + "' as " +
                    (specs[r].is_int ? "int" : "double") + " on atom line " +
                    std::to_string(i));
            }
            q = next;
        }
    }
}

class LammpsReader
{
public:
    static LammpsData
    read_lammps_atoms(const std::string& filename,
                      const std::vector<LammpsColumnSpec>& specs)
    {
        LammpsData out;

        if (filename.size() >= 3 &&
            filename.compare(filename.size() - 3, 3, ".gz") == 0)
        {
            std::string content = readGzipFile_fast_zlib(filename);
            parse_lammps_atoms_buffer(
                content.data(), content.data() + content.size(), out, specs);
            return out;
        }

        MMapBuffer mm = map_file_readonly(filename);
        if (UNLIKELY(!mm.ptr || !mm.len))
        {
            throw std::invalid_argument("LAMMPS file is empty: " + filename);
        }
        parse_lammps_atoms_buffer(mm.ptr, mm.ptr + mm.len, out, specs);
        return out;
    }
};

#endif // DATA_READ_HPP