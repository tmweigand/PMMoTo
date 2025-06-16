#ifndef BINS_H
#define BINS_H

#include <iostream>
#include <vector>

void
count_locations(const std::vector<std::vector<double> >& coordinates,
                int dimension,
                std::vector<uint64_t>& bins,
                double bin_width,
                double min_bin_value)
{
    for (const auto& coords : coordinates)
    {
        int bin_index =
            std::floor((coords[dimension] - min_bin_value) / bin_width);

        if (bin_index >= 0 && bin_index < static_cast<int>(bins.size()))
        {
            bins[bin_index]++;
        }
    }
    // return bins;
};

void
sum_masses(const std::vector<std::vector<double> >& coordinates,
           const std::vector<double>& masses,
           int dimension,
           std::vector<double>& bins,
           double bin_width,
           double min_bin_value)
{
    for (size_t i = 0; i < coordinates.size(); ++i)
    {
        const auto& coords = coordinates[i];
        double mass = masses[i];
        int bin_index =
            std::floor((coords[dimension] - min_bin_value) / bin_width);

        if (bin_index >= 0 && bin_index < static_cast<int>(bins.size()))
        {
            bins[bin_index] = bins[bin_index] + mass;
        }
    }
    // return bins;
};

#endif