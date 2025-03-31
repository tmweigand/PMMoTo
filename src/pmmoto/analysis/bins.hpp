#ifndef BINS_H
#define BINS_H

#include <vector>
#include <iostream>

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
        
        if (bin_index >= 0 && bin_index < bins.size() ){
            bins[bin_index]++;
        }
    }
    return bins;
};


std::vector<double>
sum_masses(const std::vector<std::vector<double> >& coordinates,
                const std::vector<double> & masses,
                int dimension,
                std::vector<double> bins,
                double bin_width,
                double min_bin_value)
{
    for (size_t i = 0; i < coordinates.size(); ++i)
    {
        const auto& coords = coordinates[i];
        double mass = masses[i];
        int bin_index =
            std::floor((coords[dimension] - min_bin_value) / bin_width);
         
        if (bin_index >= 0 && bin_index < bins.size() ){
            
            bins[bin_index] = bins[bin_index] + mass;
        }
    }
    return bins;
};

#endif