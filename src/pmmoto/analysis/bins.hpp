#ifndef BINS_H
#define BINS_H

#include <vector>

std::vector<unsigned long long>
count_locations(const std::vector<std::vector<double> >& coordinates,
                int dimension,
                std::vector<unsigned long long> bins,
                double bin_width,
                double min_bin_value)
{
    for (const auto& coords : coordinates)
    {
        int bin_index =
            std::floor((coords[dimension] - min_bin_value) / bin_width);
        bins[bin_index]++;
    }
    return bins;
};

#endif