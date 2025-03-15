#ifndef RDF_H
#define RDF_H

#include "atoms.hpp"

std::vector<unsigned long long>
generate_rdf(std::shared_ptr<AtomList> probe_atoms,
             std::shared_ptr<AtomList> atoms,
             double max_distance,
             std::vector<unsigned long long> bins,
             double bin_width)
{
    if (!probe_atoms)
    {
        throw std::runtime_error("probe_atoms is null!");
    }

    auto particles = probe_atoms->get_coordinates();
    for (const auto& probe_coordinates : particles)
    {
        auto distances =
            atoms->collect_kd_distances(probe_coordinates,
                                        max_distance,
                                        false); // false so not squared

        for (double distance : distances)
        {
            int bin_index = std::floor(distance / bin_width);
            bins[bin_index]++;
        }
    }
    return bins;
};

#endif