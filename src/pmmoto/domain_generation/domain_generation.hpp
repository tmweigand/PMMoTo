#ifndef DOMAINGENERATION_H
#define DOMAINGENERATION_H

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "KDTreeVectorOfVectorsAdaptor.h"
#include "nanoflann.hpp"
#include "sphere_pack.hpp"

/**
 * @struct Grid
 * @brief x,y,z coordinates of voxel centroids
 */
struct Grid {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
  std::vector<size_t> strides;
};

struct Verlet {
  size_t num_verlet;
  std::vector<std::vector<double>> centroids;
  std::vector<double> diameters;
  std::vector<std::vector<std::vector<size_t>>> loops;
};
/**
 * @brief Brute force approach
 * otherwise).
 */
void brute_force(uint8_t *img, const std::vector<double> voxel,
                 const std::vector<size_t> &strides,
                 const std::shared_ptr<SphereList> &sphere_list,
                 const std::vector<size_t> &verlet_spheres, size_t i, size_t j,
                 size_t k) {

  for (size_t index : verlet_spheres) {
    const Sphere &sphere = (*sphere_list)[index];
    if (in_sphere(voxel, sphere.coordinates, sphere.radius_squared)) {
      size_t stride = i * strides[0] + j * strides[1] + k;
      img[stride] = 0;
      break;
    }
  }
}

/**
 * @brief Determines if voxel centroids are inside any sphere.
 * @param spheres A vector of spheres, each represented as {x, y, z, r}.
 * @return A 3D vector representing the voxel grid (1 if inside any sphere, 0
 * otherwise).
 */
void gen_sphere_img_brute_force(
    uint8_t *img, const Grid &grid, Verlet verlet,
    const std::shared_ptr<SphereList> &sphere_list) {

  std::vector<double> voxel(3);

  for (size_t n = 0; n < verlet.num_verlet; ++n) {
    std::vector<std::vector<size_t>> loop = verlet.loops[n];
    std::vector<size_t> verlet_spheres = sphere_list->collect_verlet_spheres(
        verlet.diameters[n], verlet.centroids[n]);

    for (size_t i = loop[0][0]; i < loop[0][1]; ++i) {
      voxel[0] = grid.x[i];
      for (size_t j = loop[1][0]; j < loop[1][1]; ++j) {
        voxel[1] = grid.y[j];
        for (size_t k = loop[2][0]; k < loop[2][1]; ++k) {
          voxel[2] = grid.z[k];
          brute_force(img, voxel, grid.strides, sphere_list, verlet_spheres, i,
                      j, k);
        }
      }
    }
  }
}

/**
 * @brief kd approach
 */
void kd_method(uint8_t *img, const std::vector<double> &voxel, double radius,
               const std::shared_ptr<SphereList> &sphere_list,
               const std::vector<size_t> &strides, size_t i, size_t j,
               size_t k) {

  std::vector<size_t> indices = sphere_list->collect_kd_indices(voxel, radius);
  for (const auto &index : indices) {
    const Sphere &sphere = (*sphere_list)[index];
    if (in_sphere(voxel, sphere.coordinates, sphere.radius_squared)) {
      size_t stride = i * strides[0] + j * strides[1] + k;
      img[stride] = 0;
      return;
    }
  }
}

/**
 * @brief Determines if voxel centroids are inside any sphere.
 * @param spheres A vector of spheres, each represented as {x, y, z, r}.
 * @return A 3D vector representing the voxel grid (1 if inside any sphere, 0
 * otherwise).
 */
void gen_sphere_img_kd_method(uint8_t *img, const Grid &grid, Verlet verlet,
                              const std::shared_ptr<SphereList> &sphere_list) {

  auto samples = sphere_list->getAllCoordinates();

  std::vector<double> voxel(3);
  std::vector<std::vector<size_t>> loop;
  std::vector<size_t> verlet_spheres;

  double radius = sphere_list->max_radius();

  for (size_t n = 0; n < verlet.num_verlet; ++n) {
    loop = verlet.loops[n];
    verlet_spheres = sphere_list->collect_verlet_spheres(verlet.diameters[n],
                                                         verlet.centroids[n]);

    for (size_t i = loop[0][0]; i < loop[0][1]; ++i) {
      voxel[0] = grid.x[i];
      for (size_t j = loop[1][0]; j < loop[1][1]; ++j) {
        voxel[1] = grid.y[j];
        for (size_t k = loop[2][0]; k < loop[2][1]; ++k) {
          voxel[2] = grid.z[k];

          kd_method(img, voxel, radius, sphere_list, grid.strides, i, j, k);
        }
      }
    }
  }
}

#endif
