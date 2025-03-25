#ifndef DATA_READ_HPP
#define DATA_READ_HPP

#include <zlib.h>

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
    std::vector<std::vector<double> > atom_positions;
    std::vector<int> atom_types;
    std::vector<std::vector<double> > domain_data;
    double timestep;
};

class LammpsReader
{
private:
    using AtomIdMap =
        std::map<std::pair<int, double>, int>;       // type, charge -> atom_id
    static constexpr size_t BUFFER_SIZE = 64 * 1024; // 64KB buffer
    static inline std::vector<std::string> tokens;

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

    // Efficient string splitting
    static void split(const std::string& str, std::vector<std::string>& out)
    {
        out.clear();
        size_t start = 0;
        size_t end = 0;

        // Pre-reserve some space to avoid reallocations
        out.reserve(8); // Most lines have 8 or fewer tokens

        while ((end = str.find(' ', start)) != std::string::npos)
        {
            if (end > start) out.emplace_back(str.substr(start, end - start));
            start = end + 1;
            // Skip multiple spaces
            while (start < str.length() && str[start] == ' ') ++start;
        }

        if (start < str.length()) out.emplace_back(str.substr(start));
    }

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

    static int readNumAtoms(const std::string& line, LammpsData& data)
    {
        int num_atoms = std::stoi(line);
        data.atom_positions.resize(num_atoms, std::vector<double>(3, 0.0));
        data.atom_types.resize(num_atoms, 0);
        return num_atoms;
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

    static void processAtomLine(const std::string& line,
                                size_t atom_count,
                                LammpsData& data)
    {
        split(line, tokens);

        if (tokens.size() >= 8)
        {
            data.atom_types[atom_count] = std::stoi(tokens[2]);

            for (size_t i = 0; i < 3; ++i)
            {
                data.atom_positions[atom_count][i] = std::stod(tokens[i + 5]);
            }
        }
    }

    static void processAtomLineWithMap(const std::string& line,
                                       size_t atom_count,
                                       LammpsData& data,
                                       const AtomIdMap& id_map)
    {
        split(line, tokens);

        if (tokens.size() >= 8)
        {
            int type = std::stoi(tokens[2]);
            double charge = std::stod(tokens[4]);

            auto it = id_map.find({ type, charge });
            if (it == id_map.end())
            {
                throw std::runtime_error("No atom ID found for type " +
                                         std::to_string(type) + " and charge " +
                                         std::to_string(charge));
            }
            data.atom_types[atom_count] = it->second;

            // Store positions - use direct conversion from string_view
            for (size_t i = 0; i < 3; ++i)
            {
                data.atom_positions[atom_count][i] = std::stod(tokens[i + 5]);
            }
        }
    }

public:
    static LammpsData read_lammps_atoms(const std::string& filename)
    {
        LammpsData data;
        data.domain_data =
            std::vector<std::vector<double> >(3, std::vector<double>(2, 0.0));

        auto file = openFile(filename);
        std::string line;
        size_t line_count = 0;
        size_t num_atoms = 0;
        size_t atom_count = 0;

        // Pre-allocate tokens vector
        tokens.reserve(10);

        while (std::getline(file, line))
        {
            if (line.empty())
            {
                ++line_count;
                continue;
            }

            if (line_count == 1)
            {
                data.timestep = std::stod(line);
            }
            else if (line_count == 3)
            {
                num_atoms = std::stoull(line);
                data.atom_positions.resize(num_atoms,
                                           std::vector<double>(3, 0.0));
                data.atom_types.resize(num_atoms);
            }
            else if (line_count >= 5 && line_count <= 7)
            {
                readDomainBounds(line, line_count - 5, data);
            }
            else if (line_count >= 9 && atom_count < num_atoms)
            {
                processAtomLine(line, atom_count++, data);
            }
            ++line_count;
        }

        return data;
    }

    static LammpsData read_lammps_atoms_with_map(const std::string& filename,
                                                 const AtomIdMap& id_map)
    {
        LammpsData data;
        data.domain_data =
            std::vector<std::vector<double> >(3, std::vector<double>(2, 0.0));

        auto file = openFile(filename);
        std::string line;
        size_t line_count = 0;
        size_t num_atoms = 0;
        size_t atom_count = 0;

        while (std::getline(file, line))
        {
            if (line_count == 1)
            {
                data.timestep = readTimestep(line);
            }
            else if (line_count == 3)
            {
                num_atoms = readNumAtoms(line, data);
            }
            else if (line_count >= 5 && line_count <= 7)
            {
                readDomainBounds(line, line_count - 5, data);
            }
            else if (line_count >= 9 && atom_count < num_atoms)
            {
                processAtomLineWithMap(line, atom_count, data, id_map);
                atom_count++;
            }
            line_count++;
        }

        return data;
    }
};

#endif