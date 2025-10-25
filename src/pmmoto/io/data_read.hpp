#ifndef DATA_READ_HPP
#define DATA_READ_HPP

// You vendored fast_float headers under extern/fast_float/include
// Ensure setup.py adds that path to include_dirs.
#include "fast_float.h"
#include <sys/mman.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <unistd.h>
#include <zlib.h>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// Result container (must match Cython .pxd)
struct LammpsData
{
    std::vector<uint64_t> atom_ids;
    std::vector<uint8_t> atom_types;
    std::vector<double> atom_positions;            // 3N flat
    std::vector<std::vector<double> > domain_data; // [[lo,hi],[lo,hi],[lo,hi]]
    double timestep = 0.0;
};

// ====== Fast IO helpers ======

// RAII mmap view (no copy for plain files)
struct MMapBuffer
{
    const char* ptr = nullptr;
    size_t len = 0;
    void* map = nullptr;

    MMapBuffer() = default;
    MMapBuffer(const MMapBuffer&) = delete;
    MMapBuffer& operator=(const MMapBuffer&) = delete;

    // Move constructor and assignment
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
        if (map && len)
        {
            munmap(map, len);
        }
    }
};

static inline MMapBuffer
map_file_readonly(const std::string& filename)
{
    MMapBuffer out;
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open failed: " + filename);

    struct stat st
    {
    };
    if (fstat(fd, &st) != 0)
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
    if (p == MAP_FAILED)
    {
        throw std::runtime_error("mmap failed (" + std::to_string(e) +
                                 "): " + filename);
    }
    out.map = p;
    out.ptr = static_cast<const char*>(p);
    out.len = size;
    return out;
}

// Read last 4 bytes of gzip (ISIZE) to get uncompressed size (mod 2^32)
static inline size_t
gzip_uncompressed_size_from_footer(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Could not open gz file: " + filename);
    auto end = ifs.tellg();
    if (end < std::streamoff(4))
        throw std::runtime_error("Invalid gz (too small): " + filename);
    ifs.seekg(end - std::streamoff(4));
    unsigned char isize[4]{};
    ifs.read(reinterpret_cast<char*>(isize), 4);
    if (!ifs) throw std::runtime_error("Failed reading gz footer: " + filename);
    size_t n = size_t(isize[0]) | (size_t(isize[1]) << 8) |
               (size_t(isize[2]) << 16) | (size_t(isize[3]) << 24);
    return n;
}

// Single-pass zlib inflate using exact-size allocation from ISIZE
static inline std::string
readGzipFile_fast_zlib(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("open gz failed: " + filename);
    size_t comp_size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    std::string comp;
    comp.resize(comp_size);
    ifs.read(comp.data(), comp_size);
    if (!ifs) throw std::runtime_error("read gz failed: " + filename);
    ifs.close();

    size_t out_size = gzip_uncompressed_size_from_footer(filename);
    std::string out;
    out.resize(out_size);

    z_stream strm{};
    strm.next_in = reinterpret_cast<Bytef*>(comp.data());
    strm.avail_in = static_cast<uInt>(comp.size());
    strm.next_out = reinterpret_cast<Bytef*>(out.data());
    strm.avail_out = static_cast<uInt>(out.size());

    int ret = inflateInit2(&strm, 16 + MAX_WBITS);
    if (ret != Z_OK)
        throw std::runtime_error("inflateInit2 failed: " + std::to_string(ret));

    ret = inflate(&strm, Z_FINISH);
    if (ret != Z_STREAM_END)
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
            if (ret != Z_OK)
                throw std::runtime_error("inflateInit2 retry failed: " +
                                         std::to_string(ret));
            ret = inflate(&strm, Z_FINISH);
        }
    }
    if (ret != Z_STREAM_END)
    {
        inflateEnd(&strm);
        throw std::runtime_error("inflate failed: " + std::to_string(ret));
    }

    size_t actual = out.size() - strm.avail_out;
    inflateEnd(&strm);
    out.resize(actual);
    return out;
}

// ====== Pointer-based parsing helpers ======

static inline const char*
skip_spaces(const char* p, const char* end)
{
    while (p < end)
    {
        char c = *p;
        if (c != ' ' && c != '\t' && c != '\r') break;
        ++p;
    }
    return p;
}

static inline const char*
skip_token(const char* p, const char* end)
{
    while (p < end)
    {
        char c = *p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') break;
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
parse_int_ptr(const char* p, const char* end, int& out)
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
    while (p < end && *p != '\n') ++p;
    line_end = p;
    if (p < end && *p == '\n') ++p;
    if (line_end > line_start && *(line_end - 1) == '\r') --line_end;
}

// Parse one atom line: id mol type mass q x y z
static inline const char*
parse_atom_fields(const char* p,
                  const char* end,
                  uint64_t& id_out,
                  int& type_out,
                  double coords[3],
                  double& charge_out)
{
    p = skip_spaces(p, end);
    p = parse_uint64_ptr(p, end, id_out);
    p = skip_spaces(p, end);
    p = skip_token(p, end); // mol
    p = skip_spaces(p, end);
    p = parse_int_ptr(p, end, type_out);
    p = skip_spaces(p, end);
    p = skip_token(p, end); // mass
    p = skip_spaces(p, end);
    p = parse_double_ptr(p, end, charge_out);
    p = skip_spaces(p, end);
    p = parse_double_ptr(p, end, coords[0]);
    p = skip_spaces(p, end);
    p = parse_double_ptr(p, end, coords[1]);
    p = skip_spaces(p, end);
    p = parse_double_ptr(p, end, coords[2]);
    p = skip_spaces(p, end);
    return p;
}

// ====== Main parser ======

struct PairHash
{
    std::size_t operator()(const std::pair<int, double>& p) const noexcept
    {
        uint64_t dbits;
        static_assert(sizeof(double) == sizeof(uint64_t),
                      "double must be 64-bit");
        std::memcpy(&dbits, &p.second, sizeof(dbits));
        std::size_t h1 = std::hash<int>{}(p.first);
        std::size_t h2 = std::hash<uint64_t>{}(dbits);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

static inline void
parse_lammps_atoms_buffer(
    const char* buf,
    const char* end,
    LammpsData& data,
    const std::map<std::pair<int, double>, int>* id_map = nullptr)
{
    const char* p = buf;
    const char* line_s = nullptr;
    const char* line_e = nullptr;

    read_line(p, end, line_s, line_e); // "ITEM: TIMESTEP"
    read_line(p, end, line_s, line_e); // timestep value
    {
        double ts = 0.0;
        parse_double_ptr(line_s, line_e, ts);
        data.timestep = ts;
    }

    read_line(p, end, line_s, line_e); // "ITEM: NUMBER OF ATOMS"
    read_line(p, end, line_s, line_e); // number
    uint64_t natoms_u = 0;
    parse_uint64_ptr(line_s, line_e, natoms_u);
    size_t natoms = static_cast<size_t>(natoms_u);

    data.atom_ids.resize(natoms);
    data.atom_types.resize(natoms);
    data.atom_positions.resize(3 * natoms);
    data.domain_data.resize(3, std::vector<double>(2));

    read_line(p, end, line_s, line_e); // "ITEM: BOX BOUNDS ..."
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

    read_line(p, end, line_s, line_e); // "ITEM: ATOMS ..."

    std::unordered_map<std::pair<int, double>, int, PairHash> fast_map;
    if (id_map && !id_map->empty())
    {
        fast_map.reserve(id_map->size() * 2);
        for (const auto& kv : *id_map)
        {
            fast_map.emplace(kv.first, kv.second);
        }
    }

    size_t i = 0;
    while (i < natoms && p < end)
    {
        uint64_t id = 0;
        int type = 0;
        double q = 0.0;
        double xyz[3] = { 0.0, 0.0, 0.0 };

        const char* after = parse_atom_fields(p, end, id, type, xyz, q);

        data.atom_ids[i] = id;
        if (!fast_map.empty())
        {
            auto it = fast_map.find({ type, q });
            if (it == fast_map.end())
            {
                throw std::runtime_error("No mapping for type/charge: type=" +
                                         std::to_string(type));
            }
            data.atom_types[i] = static_cast<uint8_t>(it->second);
        }
        else
        {
            data.atom_types[i] = static_cast<uint8_t>(type);
        }
        size_t base = 3 * i;
        data.atom_positions[base + 0] = xyz[0];
        data.atom_positions[base + 1] = xyz[1];
        data.atom_positions[base + 2] = xyz[2];

        ++i;
        p = after;
        while (p < end && *p != '\n') ++p;
        if (p < end && *p == '\n') ++p;
    }
}

// Wrap in LammpsReader class to match Cython interface
class LammpsReader
{
public:
    using AtomIdMap = std::map<std::pair<int, double>, int>;

    static LammpsData read_lammps_atoms(const std::string& filename,
                                        const AtomIdMap* id_map = nullptr)
    {
        LammpsData out;
        if (filename.size() >= 3 &&
            filename.compare(filename.size() - 3, 3, ".gz") == 0)
        {
            std::string content = readGzipFile_fast_zlib(filename);
            parse_lammps_atoms_buffer(
                content.data(), content.data() + content.size(), out, id_map);
            return out;
        }

        MMapBuffer mm = map_file_readonly(filename);
        if (mm.ptr && mm.len)
        {
            parse_lammps_atoms_buffer(mm.ptr, mm.ptr + mm.len, out, id_map);
        }
        return out;
    }
};

#endif // DATA_READ_HPP