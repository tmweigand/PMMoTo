#ifndef KDTREE_H
#define KDTREE_H

#include <iostream>
#include <stdexcept>
#include <vector>

#include "KDTreeVectorOfVectorsAdaptor.h"
#include "nanoflann.hpp"
// #include "utils.h"

using KDTreeType =
    KDTreeVectorOfVectorsAdaptor<std::vector<std::vector<double>>, double>;

class KDTree {
private:
  std::unique_ptr<KDTreeType> tree; // Store tree in a smart pointer
  std::shared_ptr<std::vector<std::vector<double>>>
      data_ref; // Shared ownership

public:
  KDTree() : tree(nullptr) {} // Constructor initializes tree pointer to nullptr

  void initialize_kd(std::shared_ptr<std::vector<std::vector<double>>> data,
                     int max_leaf_size = 10) {
    if (!data || data->empty()) {
      throw std::runtime_error("Cannot create KD-tree with empty input.");
    }

    // Store shared pointer
    data_ref = data;
    // Create KD-tree and assign it to the member variable
    tree = std::make_unique<KDTreeType>(3 /*dim*/, *data_ref, max_leaf_size);
  }

  KDTreeType *getTree() {
    return tree.get(); // Provide access to the tree
  }

  void check_tree() {
    if (!tree) {
      throw std::runtime_error(
          "KDTree is not initialized. Call initialize_kd() first.");
    }

    if (!tree->index) {
      throw std::runtime_error("KDTree index is not initialized.");
    }
  }

  // Collect indices and/or distance for points in radius
  std::vector<size_t> radius_search_indices(const std::vector<double> &voxel,
                                            double radius) {

    check_tree();

    std::vector<nanoflann::ResultItem<size_t, double>> ret_matches;
    const size_t nMatches =
        tree->index->radiusSearch(&voxel[0], radius, ret_matches);

    std::vector<size_t> indices;
    indices.reserve(nMatches);
    for (const auto &match : ret_matches) {
      indices.emplace_back(match.first);
    }

    return indices;
  }

  // Collect indices and/or distance for points in radius
  std::vector<double> radius_search_distances(const std::vector<double> &voxel,
                                              double radius) {

    check_tree();

    std::vector<nanoflann::ResultItem<size_t, double>> ret_matches;
    const size_t nMatches =
        tree->index->radiusSearch(&voxel[0], radius, ret_matches);

    std::vector<double> distances;
    distances.reserve(nMatches);
    for (const auto &match : ret_matches) {
      distances.emplace_back(match.second);
    }

    return distances;
  }
};

#endif