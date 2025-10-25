#ifndef DATA_READ_HPP
#define DATA_READ_HPP

#include "fast_float.h"

#include <zlib.h>

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct LammpsData
{
    std::vector<double> atom_positions;
    std::vector<uint64_t> atom_ids;
    std::vector<uint8_t> atom_types;
    std::vector<std::vector<double> > domain_data;
    double timestep = 0.0;
};

class LammpsReader
{
private:
    using AtomIdMap =
        std::map<std::pair<int, double>, int>;       // type, charge -> atom_id
    static constexpr size_t BUFFER_SIZE = 64 * 1024; // 64KB buffer

    class GzFileWrapper
    {
        gzFile file_;

    public:
        explicit GzFileWrapper(const std::string& filename)
        {
            file_ = gzopen(filename.c_str(), "r");
            if (!file_)
            {
                throw std::runtime_error("Could not open gzipped file: " +
                                         filename);
            }
        }

        ~GzFileWrapper()
        {
            if (file_)
            {
                gzclose(file_);
            }
        }

        gzFile get()
        {
            return file_;
        }

        // Prevent copying
        GzFileWrapper(const GzFileWrapper&) = delete;
        GzFileWrapper& operator=(const GzFileWrapper&) = delete;
    };

    static std::string readGzipFile(const std::string& filename)
    {
        GzFileWrapper file(filename); // RAII wrapper handles cleanup

        std::string content;
        content.reserve(1024 * 1024);
        std::vector<char> buffer(BUFFER_SIZE);

        while (int bytesRead = gzread(file.get(), buffer.data(), buffer.size()))
        {
            if (bytesRead < 0)
            {
                throw std::runtime_error("Error reading gzipped file: " +
                                         filename);
            }
            content.append(buffer.data(), bytesRead);
        }

        return content;
    }

    static std::string readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file)
        {
            throw std::runtime_error("Could not open file: " + filename);
        }

        // Get file size and reserve space
        size_t size = file.tellg();
        file.seekg(0);

        std::string content;
        content.reserve(size);
        content.assign(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());

        file.close();
        return content;
    }

    static std::stringstream openFile(const std::string& filename)
    {
        std::string content;
        // C++17 compatible check for .gz extension
        if (filename.length() > 3 &&
            filename.compare(filename.length() - 3, 3, ".gz") == 0)
        {
            content = readGzipFile(filename);
        }
        else
        {
            content = readFile(filename);
        }
        return std::stringstream(std::move(content));
    }

    static double readTimestep(const std::string& line)
    {
        return std::stod(line);
    }

    static void
    readDomainBounds(const std::string& line, int dim, LammpsData& data)
    {
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos)
        {
            data.domain_data[dim][0] = std::stod(line.substr(0, space_pos));
            data.domain_data[dim][1] = std::stod(line.substr(space_pos + 1));
        }
    }

    static inline const char* skip_spaces(const char* p, const char* end)
    {
        while (p < end && (*p == ' ' || *p == '\t')) ++p;
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
        // fast_float::from_chars is much faster than std::stod/strtod
        auto result = fast_float::from_chars(p, end, out);
        return result.ptr;
    }

    // Replace processAtomLine with a pointer-based routine used in the main
    // loop
    static inline const char* parse_atom_fields(const char* p,
                                                const char* end,
                                                uint64_t& id_out,
                                                int& type_out,
                                                double coords[3],
                                                double& charge_out,
                                                bool parse_charge)
    {
        p = skip_spaces(p, end);
        p = parse_uint64_ptr(p, end, id_out);
        p = skip_spaces(p, end);

        // mol (skip token)
        while (p < end && *p != ' ' && *p != '\t' && *p != '\n') ++p;
        p = skip_spaces(p, end);

        p = parse_int_ptr(p, end, type_out);
        p = skip_spaces(p, end);

        // mass (skip token)
        while (p < end && *p != ' ' && *p != '\t' && *p != '\n') ++p;
        p = skip_spaces(p, end);

        if (parse_charge)
        {
            p = parse_double_ptr(p, end, charge_out);
            p = skip_spaces(p, end);
        }
        else
        {
            // skip q token
            while (p < end && *p != ' ' && *p != '\t' && *p != '\n') ++p;
            p = skip_spaces(p, end);
        }

        for (int i = 0; i < 3; ++i)
        {
            p = parse_double_ptr(p, end, coords[i]);
            p = skip_spaces(p, end);
        }
        return p;
    }

    static inline void processAtomLine(const std::string_view& line,
                                       size_t atom_count,
                                       LammpsData& data)
    {
        size_t pos = 0;
        size_t next;

        // Parse id
        next = line.find(' ', pos);
        data.atom_ids[atom_count] =
            std::stoull(std::string(line.substr(pos, next - pos)));
        pos = next + 1;

        // Parse mol (skip)
        next = line.find(' ', pos);
        pos = next + 1;

        // Parse type
        next = line.find(' ', pos);
        data.atom_types[atom_count] =
            std::stoi(std::string(line.substr(pos, next - pos)));
        pos = next + 1;

        // Parse mass (skip)
        next = line.find(' ', pos);
        pos = next + 1;

        // Parse q (skip)
        next = line.find(' ', pos);
        pos = next + 1;

        // Parse x, y, z
        for (int i = 0; i < 3; ++i)
        {
            next = line.find(' ', pos);
            data.atom_positions[3 * atom_count + i] =
                std::stod(std::string(line.substr(pos, next - pos)));
            pos = next + 1;
        }
    }

    static inline void processAtomLine(const std::string_view& line,
                                       size_t atom_count,
                                       LammpsData& data,
                                       const AtomIdMap& id_map)
    {
        size_t pos = 0;
        size_t next;

        // Parse id
        next = line.find(' ', pos);
        data.atom_ids[atom_count] =
            std::stoull(std::string(line.substr(pos, next - pos)));
        pos = next + 1;

        // Parse mol (skip)
        next = line.find(' ', pos);
        pos = next + 1;

        // Parse type
        next = line.find(' ', pos);
        auto type = std::stoi(std::string(line.substr(pos, next - pos)));
        pos = next + 1;

        // Parse mass (skip)
        next = line.find(' ', pos);
        pos = next + 1;

        // Parse q - charge
        next = line.find(' ', pos);
        auto charge = std::stod(std::string(line.substr(pos, next - pos)));
        pos = next + 1;

        auto it = id_map.find({ type, charge });
        if (it == id_map.end())
        {
            throw std::runtime_error("No atom ID found for type " +
                                     std::to_string(type) + " and charge " +
                                     std::to_string(charge));
        }

        data.atom_types[atom_count] = it->second;

        // Parse x, y, z
        for (int i = 0; i < 3; ++i)
        {
            next = line.find(' ', pos);
            data.atom_positions[3 * atom_count + i] =
                std::stod(std::string(line.substr(pos, next - pos)));
            pos = next + 1;
        }
    }

public:
    static LammpsData read_lammps_atoms(const std::string& filename,
                                        const AtomIdMap* id_map = nullptr)
    {
        LammpsData data;
        data.domain_data = { { { 0.0, 0.0 }, { 0.0, 0.0 }, { 0.0, 0.0 } } };

        // Read whole file (gz or plain) into a single string buffer
        std::string content;
        if (filename.size() > 3 &&
            filename.compare(filename.size() - 3, 3, ".gz") == 0)
        {
            content = readGzipFile(filename);
        }
        else
        {
            content = readFile(filename);
        }

        const char* buf = content.data();
        const char* end = buf + content.size();
        const char* p = buf;

        // helper to read a single line [start, end) and advance p to next line
        auto read_line = [&](const char*& start, const char*& line_end)
        {
            start = p;
            while (p < end && *p != '\n') ++p;
            line_end = p;
            if (p < end && *p == '\n') ++p; // skip newline
            // trim optional '\r' at end
            if (line_end > start && *(line_end - 1) == '\r') --line_end;
        };

        const char* line_start;
        const char* line_end;

        // Skip "ITEM: TIMESTEP"
        read_line(line_start, line_end);

        // Timestep line -> parse double
        read_line(line_start, line_end);
        {
            double ts = 0.0;
            parse_double_ptr(line_start, line_end, ts);
            data.timestep = ts;
        }

        // Skip "ITEM: NUMBER OF ATOMS"
        read_line(line_start, line_end);

        // Number of atoms
        read_line(line_start, line_end);
        uint64_t num_atoms_u = 0;
        parse_uint64_ptr(line_start, line_end, num_atoms_u);
        size_t num_atoms = static_cast<size_t>(num_atoms_u);

        data.atom_positions.resize(3 * num_atoms, 0.0);
        data.atom_ids.resize(num_atoms);
        data.atom_types.resize(num_atoms);

        // Skip "ITEM: BOX BOUNDS ..."
        read_line(line_start, line_end);

        // Read 3 domain lines, each with two doubles
        for (int i = 0; i < 3; ++i)
        {
            read_line(line_start, line_end);
            const char* q = line_start;
            double lo = 0.0, hi = 0.0;
            q = parse_double_ptr(q, line_end, lo);
            q = skip_spaces(q, line_end);
            q = parse_double_ptr(q, line_end, hi);
            data.domain_data[i][0] = lo;
            data.domain_data[i][1] = hi;
        }

        // Skip "ITEM: ATOMS id mol type mass q x y z ..."
        read_line(line_start, line_end);

        // Now parse atom lines in a tight pointer loop
        size_t atom_count = 0;
        while (atom_count < num_atoms && p < end)
        {
            uint64_t id = 0;
            int type = 0;
            double coords[3] = { 0.0, 0.0, 0.0 };
            double charge = 0.0;

            // parse_atom_fields advances p to after the coordinates
            p = parse_atom_fields(
                p, end, id, type, coords, charge, id_map != nullptr);

            data.atom_ids[atom_count] = id;

            if (id_map)
            {
                auto it = id_map->find({ type, charge });
                if (it == id_map->end())
                {
                    throw std::runtime_error(
                        "No atom ID found for type " + std::to_string(type) +
                        " and charge " + std::to_string(charge));
                }
                data.atom_types[atom_count] = static_cast<uint8_t>(it->second);
            }
            else
            {
                data.atom_types[atom_count] = static_cast<uint8_t>(type);
            }

            const size_t base = 3 * atom_count;
            data.atom_positions[base + 0] = coords[0];
            data.atom_positions[base + 1] = coords[1];
            data.atom_positions[base + 2] = coords[2];

            ++atom_count;

            // advance p to the start of next line (skip remainder of this line)
            while (p < end && *p != '\n') ++p;
            if (p < end && *p == '\n') ++p;
        }

        return data;
    }
};

#endif