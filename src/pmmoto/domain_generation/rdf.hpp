#ifndef RDF_H
#define RDF_H

#include "atoms.hpp"

std::vector<long int> _generate_rdf(std::shared_ptr<AtomList> probe_atoms,
                                    std::shared_ptr<AtomList> atoms,
                                    double maximum_distance, int num_bins) {

  double bin_width = maximum_distance / num_bins;
  std::vector<long int> bins(num_bins, 0);

  for (const auto &probe_atom : probe_atoms->atoms) {
    auto distances =
        atoms->collect_kd_distances(probe_atom.coordinates, maximum_distance);
    for (double distance : distances) {
      int bin_index =
          std::floor(distance / bin_width); // Determine the correct bin index
      bins[bin_index]++;                    // Increment the count for that bin
    }
  }
  return bins;
};

#endif