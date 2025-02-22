#ifndef SPHEREPACK_H
#define SPHEREPACK_H

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "kd_tree.hpp"

/**
 * @struct Coords
 * @brief x,y,z coordinates
 */
struct Coords {
  double x;
  double y;
  double z;
};

/**
 * @class Sphere
 * @brief Represents a sphere with position and radius.
 */
class Sphere {
public:
  Coords coordinates; // Stores x, y, z
  double radius;
  double radius_squared;

  Sphere(double x, double y, double z, double radius)
      : coordinates{x, y, z}, radius(radius), radius_squared(radius * radius) {}

  Sphere(std::vector<double> coordinates, double radius)
      : coordinates{coordinates[0], coordinates[1], coordinates[2]},
        radius(radius), radius_squared(radius * radius) {}
};

/**
 * @brief Checks if a point is inside a sphere.
 *
 * @param point The point as {x, y, z}.
 * @param sphere_center The sphere center as {x, y, z}.
 * @param s_r Squared radius of the sphere.
 * @return 1 if the point is inside the sphere, 0 otherwise.
 */
inline uint8_t in_sphere(const std::vector<double> &point, const Sphere &sphere,
                         double s_r) noexcept {
  double dx = sphere.coordinates.x - point[0];
  double dy = sphere.coordinates.y - point[1];
  double dz = sphere.coordinates.z - point[2];
  return (dx * dx + dy * dy + dz * dz) <= s_r ? 1 : 0;
}

/**
 * @class SphereList
 * @brief Contains a list of the spheres and tools to operate on those spheres
 */
class SphereList {

private:
  std::vector<Sphere> spheres;

  void initializeFromVectors(std::vector<std::vector<double>> sphere_data,
                             bool build_kd) {
    spheres.reserve(sphere_data.size());

    for (auto &data : sphere_data) {
      if (data.size() < 4) {
        throw std::invalid_argument("Each sphere entry must have exactly 4 "
                                    "elements (x, y, z, radius).");
      }
      spheres.emplace_back(data[0], data[1], data[2], data[3]);
    }

    if (build_kd) {
      initializeKDTree();
    }
  }

  void initializeFromSpheres(std::vector<Sphere> sphere_data, bool build_kd) {
    spheres = std::move(sphere_data);

    if (build_kd) {
      initializeKDTree();
    }
  }

  void initializeKDTree() {
    std::vector<std::vector<double>> coords;
    coords.reserve(spheres.size());

    for (const auto &sphere : spheres) {
      coords.push_back(
          {sphere.coordinates.x, sphere.coordinates.y, sphere.coordinates.z});
    }

    kd_tree.initialize_kd(
        std::make_shared<std::vector<std::vector<double>>>(std::move(coords)));
  }

public:
  KDTree kd_tree;

  // Constructor
  // SphereList(std::vector<std::vector<double>> &sphere_data,
  //            bool build_kd = false) {

  //   for (const auto &data : sphere_data) {
  //     spheres.emplace_back(data[0], data[1], data[2], data[3]);
  //   }

  //   if (build_kd) {
  //     std::vector<std::vector<double>> coord = getAllCoordinates();
  //     auto data = std::make_shared<std::vector<std::vector<double>>>(coord);
  //     kd_tree.initialize_kd(data);
  //   }
  // }

  SphereList(std::vector<std::vector<double>> sphere_data,
             bool build_kd = false) {
    initializeFromVectors(std::move(sphere_data), build_kd);
  }

  // Constructor that accepts vector of Sphere objects
  SphereList(std::vector<Sphere> sphere_data, bool build_kd = false) {
    initializeFromSpheres(std::move(sphere_data), build_kd);
  }

  // Get the number of spheres stored
  size_t size() const { return spheres.size(); }

  // Overload operator[]
  const Sphere &operator[](size_t index) const {
    if (index >= size()) {
      throw std::out_of_range("Index out of bounds!");
    }
    return spheres[index];
  }

  /**
   * @brief Determine the largest radius in the list
   * @return the maximum radius in the list of spheres
   * radius.
   */
  double max_radius() const {
    return std::max_element(spheres.begin(), spheres.end(),
                            [](const Sphere &a, const Sphere &b) {
                              return a.radius < b.radius;
                            })
        ->radius;
  }

  /**
   * @brief Collect the coordinates for all spheres
   * @return A vector of the sphere coordinates
   * radius.
   */
  std::vector<std::vector<double>> getAllCoordinates() const {
    std::vector<std::vector<double>> coords;
    for (const auto &sphere : spheres) {
      coords.push_back(
          {sphere.coordinates.x, sphere.coordinates.y, sphere.coordinates.z});
    }
    return coords;
  }

  /**
   * @brief Generates a Verlet list of spheres within a given radius.
   *
   * @param verlet_radius The Verlet radius to consider.
   * @param point The reference point as {x, y, z}.
   * @param spheres A vector of spheres, each represented as {x, y, z, r}.
   * @return A vector of VerletSphere objects that are within the Verlet
   * radius.
   */
  std::vector<size_t> collect_verlet_spheres(double verlet_radius,
                                             const std::vector<double> &point) {

    std::vector<size_t> verlet_spheres;
    for (size_t i = 0; i < size(); ++i) {
      const auto &sphere = spheres[i];
      double r_squared =
          (sphere.radius + verlet_radius) * (sphere.radius + verlet_radius);

      if (in_sphere(point, sphere, r_squared)) {
        verlet_spheres.push_back(i);
      }
    }

    return verlet_spheres;
  }

  /**
   * @brief Collects spheres within a given Verlet radius and returns them as
   * Sphere objects.
   *
   * @param verlet_radius The Verlet radius to consider.
   * @param point The reference point as {x, y, z}.
   * @return A vector of Sphere objects within the Verlet radius.
   */
  std::vector<Sphere>
  collect_verlet_sphere_objects(double verlet_radius,
                                const std::vector<double> &point) {
    std::vector<Sphere> verlet_spheres;
    for (const auto &sphere : spheres) {
      double r_squared =
          (sphere.radius + verlet_radius) * (sphere.radius + verlet_radius);

      if (in_sphere(point, sphere, r_squared)) {
        verlet_spheres.push_back(sphere);
      }
    }
    return verlet_spheres;
  }

  /**
   * @brief Determine the spheres within a radius using a kd tree
   *
   * @param point The reference point as {x, y, z}.
   * @param radius the search radius
   * @return A vector of Sphere indices within the radius.
   */
  std::vector<size_t> collect_kd_spheres(const std::vector<double> &point,
                                         double radius) {

    std::vector<size_t> indices = kd_tree.radius_search_indices(point, radius);
    return indices;
  }

  /**
   * @brief Determine the spheres within a radius using a kd tree
   * and return the spheres
   * @param point The reference point as {x, y, z}.
   * @param radius the search radius
   * @return A vector of Sphere indices within the radius.
   */
  std::vector<Sphere>
  collect_kd_spheres_objects(const std::vector<double> &point, double radius) {

    std::vector<Sphere> local_spheres;
    std::vector<size_t> indices = kd_tree.radius_search_indices(point, radius);
    for (const auto &index : indices) {
      local_spheres.push_back(spheres[index]);
    }
    return local_spheres;
  }
};

/**
 * @brief Initializes a SphereList, optionally trims it using KD-tree or Verlet,
 * and returns a unique pointer to the SphereList.
 *
 * @param sphere_data A vector of spheres, each represented as {x, y, z, r}.
 * @param point The reference point as {x, y, z}.
 * @param radius The search radius for filtering spheres.
 * @param kd_tree Flag to use KD-tree or Verlet algorithm for filtering.
 * @param trim Flag to specify whether to filter the spheres using
 * trim_sphere_list.
 * @return A unique pointer to the initialized (and possibly trimmed)
 * SphereList.
 */
std::unique_ptr<SphereList>
initialize_sphere_list(std::vector<std::vector<double>> &sphere_data,
                       const std::vector<double> &point = {0.0, 0.0, 0.0},
                       double radius = 0, bool kd_tree = false,
                       bool trim = false) {

  auto sphere_list = std::make_unique<SphereList>(sphere_data, kd_tree);

  if (trim) {

    if (radius <= 0) {
      throw std::invalid_argument("Radius must be greater than zero.");
    }

    std::vector<Sphere> trimmed_spheres;
    if (kd_tree) {
      trimmed_spheres = sphere_list->collect_kd_spheres_objects(point, radius);
    } else {
      trimmed_spheres =
          sphere_list->collect_verlet_sphere_objects(radius, point);
    }
    // Reinitialize the SphereList with the filtered spheres
    sphere_list = std::make_unique<SphereList>(trimmed_spheres, kd_tree);
  }

  return sphere_list;
}

/**
 * @brief Initializes a SphereList, filters spheres using KD-tree, and
 * reinitializes with the result.
 *
 * @param sphere_data A vector of spheres, each represented as {x, y, z, r}.
 * @param point The reference point as {x, y, z}.
 * @param radius The search radius for filtering spheres.
 * @return A new SphereList containing only the spheres within the given radius.
 */
std::unique_ptr<SphereList>
trim_sphere_list(std::vector<std::vector<double>> &sphere_data,
                 const std::vector<double> &point, double radius,
                 bool kd_tree = false) {
  // Step 1: Initialize SphereList with the original sphere data and build
  // KD-tree
  SphereList sphere_list(sphere_data, kd_tree);

  std::vector<Sphere> filtered_spheres;
  if (kd_tree) {
    filtered_spheres = sphere_list.collect_kd_spheres_objects(point, radius);
  } else {
    filtered_spheres = sphere_list.collect_verlet_sphere_objects(radius, point);
  }

  return std::make_unique<SphereList>(filtered_spheres, kd_tree);
}

/**
 * @brief Initializes a SphereList, filters spheres using KD-tree, and
 * reinitializes with the result.
 *
 * @param sphere_data A vector of spheres, each represented as {x, y, z, r}.
 * @param point The reference point as {x, y, z}.
 * @param radius The search radius for filtering spheres.
 * @return A new SphereList containing only the spheres within the given radius.
 */
std::vector<Sphere>
trim_sphere_list_spheres(std::vector<std::vector<double>> &sphere_data,
                         const std::vector<double> &point, double radius,
                         bool kd_tree = false) {
  // Step 1: Initialize SphereList with the original sphere data and build
  // KD-tree
  SphereList sphere_list(sphere_data, kd_tree);

  std::vector<Sphere> filtered_spheres;
  if (kd_tree) {
    filtered_spheres = sphere_list.collect_kd_spheres_objects(point, radius);
  } else {
    filtered_spheres = sphere_list.collect_verlet_sphere_objects(radius, point);
  }

  return filtered_spheres;
}

#endif