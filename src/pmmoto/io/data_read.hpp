#ifndef DATA_READ_HPP
#define DATA_READ_HPP

#include <zlib.h>

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// struct LammpsData
// {
//     std::vector<std::vector<double> > atom_positions;
//     std::vector<uint8_t> atom_types;
//     std::vector<uint64_t> atom_ids;
//     std::vector<std::vector<double> > domain_data;
//     double timestep;
// };

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

        auto file = openFile(filename);
        std::string line;

        // Skip line ITEM: TIMESTEP
        std::getline(file, line);

        // timestep
        std::getline(file, line);
        data.timestep = std::stod(line);

        // skip line ITEM: NUMBER OF ATOMS
        std::getline(file, line);

        // num atoms
        std::getline(file, line);

        size_t num_atoms = std::stoull(line);
        data.atom_positions.resize(3 * num_atoms, 0.0);
        data.atom_ids.resize(num_atoms);
        data.atom_types.resize(num_atoms);

        // skip line ITEM: BOX BOUNDS pp pp pp
        std::getline(file, line);

        // read domain bounds (3 lines)
        for (int i = 0; i < 3; ++i)
        {
            std::getline(file, line);
            readDomainBounds(line, i, data);
        }

        // skip line ITEM: ATOMS id mol type mass q x y z
        std::getline(file, line);

        size_t atom_count = 0;
        while (atom_count < num_atoms && std::getline(file, line))
        {
            if (id_map)
                processAtomLine(line, atom_count, data, *id_map);
            else
                processAtomLine(line, atom_count, data);
            ++atom_count;
        }

        return data;
    }
};

#endif