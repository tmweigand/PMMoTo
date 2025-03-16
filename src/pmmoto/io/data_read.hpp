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
    using AtimIdMap =
        std::map<std::pair<int, double>, int>; // type, charge -> atim_id
    static inline std::vector<std::string_view> tokens;

    // Efficient string splitting
    static void split(std::string_view str,
                      std::vector<std::string_view>& tokens)
    {
        tokens.clear();
        size_t start = 0;
        size_t end = 0;

        while ((end = str.find(' ', start)) != std::string_view::npos)
        {
            if (end > start) tokens.push_back(str.substr(start, end - start));
            start = end + 1;
        }

        if (start < str.length()) tokens.push_back(str.substr(start));
    }

    static std::string readGzipFile(const std::string& filename)
    {
        gzFile file = gzopen(filename.c_str(), "r");
        if (!file)
        {
            throw std::runtime_error("Could not open gzipped file: " +
                                     filename);
        }

        // Pre-allocate buffer based on file size
        unsigned int uncompressed_size = 0;
        if (gzseek(file, 0, SEEK_END) != -1)
        {
            uncompressed_size = gztell(file);
            gzrewind(file);
        }

        std::string content;
        content.reserve(uncompressed_size > 0 ? uncompressed_size :
                                                1024 * 1024);

        std::vector<char> buffer(64 * 1024); // Larger buffer

        while (int bytesRead = gzread(file, buffer.data(), buffer.size()))
        {
            if (bytesRead < 0)
            {
                gzclose(file);
                throw std::runtime_error("Error reading gzipped file: " +
                                         filename);
            }
            content.append(buffer.data(), bytesRead);
        }

        gzclose(file);
        return content;
    }

    static std::string readFile(const std::string& filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Could not open file: " + filename);
        }
        return std::string(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
    }

    static std::stringstream openFile(const std::string& filename)
    {
        std::string content;
        if (filename.ends_with(".gz"))
        {
            content = readGzipFile(filename);
        }
        else
        {
            content = readFile(filename);
        }
        return std::stringstream(content);
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

    static void
    processAtomLine(std::string_view line, int atom_count, LammpsData& data)
    {
        split(line, tokens);

        if (tokens.size() >= 8)
        {
            data.atom_types[atom_count] = std::stoi(std::string(tokens[2]));

            for (int i = 0; i < 3; ++i)
            {
                data.atom_positions[atom_count][i] =
                    std::stod(std::string(tokens[i + 5]));
            }
        }
    }

    static void processAtomLineWithMap(std::string_view line,
                                       int atom_count,
                                       LammpsData& data,
                                       const AtimIdMap& id_map)
    {
        split(line, tokens);

        if (tokens.size() >= 8)
        {
            int type = std::stoi(std::string(tokens[2]));
            double charge = std::stod(std::string(tokens[4]));

            // Look up the ATIM ID from the map using compile-time pair creation
            auto it = id_map.find({ type, charge });
            if (it == id_map.end())
            {
                throw std::runtime_error("No ATIM ID found for type " +
                                         std::to_string(type) + " and charge " +
                                         std::to_string(charge));
            }
            data.atom_types[atom_count] = it->second;

            // Store positions - use direct conversion from string_view
            for (int i = 0; i < 3; ++i)
            {
                data.atom_positions[atom_count][i] =
                    std::stod(std::string(tokens[i + 5]));
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
        int line_count = 0;
        int num_atoms = 0;
        int atom_count = 0;

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
                processAtomLine(line, atom_count, data);
                atom_count++;
            }
            line_count++;
        }

        return data;
    }

    static LammpsData read_lammps_atoms_with_map(const std::string& filename,
                                                 const AtimIdMap& id_map)
    {
        LammpsData data;
        data.domain_data =
            std::vector<std::vector<double> >(3, std::vector<double>(2, 0.0));

        auto file = openFile(filename);
        std::string line;
        int line_count = 0;
        int num_atoms = 0;
        int atom_count = 0;

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