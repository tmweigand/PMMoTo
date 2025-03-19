#ifndef KDTREE_H
#define KDTREE_H

#include "KDTreeVectorOfVectorsAdaptor.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "nanoflann.hpp"
// #include "utils.h"

using KDTreeType =
    KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double> >, double>;

class KDTree
{
private:
    std::shared_ptr<KDTreeType> tree;
    std::shared_ptr<std::vector<std::vector<double> > > data_ref;

public:
    KDTree() : tree(nullptr)
    {
    }

    void initialize_kd(std::shared_ptr<std::vector<std::vector<double> > > data,
                       int max_leaf_size = 10)
    {
        if (!data || data->empty())
        {
            throw std::runtime_error("Cannot create KD-tree with empty input.");
        }

        data_ref = data;
        tree =
            std::make_shared<KDTreeType>(3 /*dim*/, *data_ref, max_leaf_size);
        tree->index->buildIndex();
    }

    KDTreeType* getTree()
    {
        return tree.get();
    }

    /**
     * @brief Ensure tree is initalized
     */
    void check_tree()
    {
        if (!tree)
        {
            throw std::runtime_error(
                "KDTree is not initialized. Call initialize_kd() first.");
        }

        if (!tree->index)
        {
            throw std::runtime_error("KDTree index is not initialized.");
        }
    }

    /**
     * @brief Collect indices and/or distance for points in radius
     *
     * \note Search radius and all returned distances
     *       are actually squared distances.
     */
    std::vector<size_t> radius_search_indices(const std::vector<double>& voxel,
                                              double radius)
    {
        check_tree();

        std::vector<nanoflann::ResultItem<size_t, double> > ret_matches;
        const size_t nMatches =
            tree->index->radiusSearch(&voxel[0], radius * radius, ret_matches);

        std::vector<size_t> indices;
        indices.reserve(nMatches);
        for (const auto& match : ret_matches)
        {
            indices.emplace_back(match.first);
        }

        return indices;
    }

    /**
     * @brief Collect distance for points in specified radius
     *
     * \note Search radius and all returned distances
     *       are actually squared distances.
     */
    std::vector<double>
    radius_search_distances(const std::vector<double>& voxel,
                            double radius,
                            bool return_square = true)
    {
        check_tree();

        std::vector<nanoflann::ResultItem<size_t, double> > ret_matches;
        const size_t nMatches =
            tree->index->radiusSearch(&voxel[0], radius * radius, ret_matches);

        std::vector<double> distances;
        distances.reserve(nMatches);
        if (return_square)
        {
            for (const auto& match : ret_matches)
            {
                distances.emplace_back(match.second);
            }
        }
        else
        {
            for (const auto& match : ret_matches)
            {
                distances.emplace_back(sqrt(match.second));
            }
        }

        return distances;
    }
};

#endif