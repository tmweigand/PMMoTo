#ifndef PARTICLELIST_H
#define PARTICLELIST_H

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "kd_tree.hpp"

// Type alias for Coords
using Coords = std::vector<double>;

// Particle class that holds coordinates and radius (as a vector)
struct Particle {
  Coords coordinates;
  double radius; // Single value for the radius

  // Constructor to initialize coordinates and radius
  Particle(Coords c = {0.0, 0.0, 0.0}, double r = 0.0)
      : coordinates(c), radius(r) {}
  Particle(double x = 0.0, double y = 0.0, double z = 0.0, double r = 0.0)
      : coordinates({x, y, z}), radius(r) {}
};

/**
 * @brief Checks if a point is inside a sphere.
 *
 * @param point The point as {x, y, z}.
 * @param sphere_center The sphere center as {x, y, z}.
 * @param s_r Squared radius of the sphere.
 * @return 1 if the point is inside the sphere, 0 otherwise.
 */
inline uint8_t in_sphere(const std::vector<double> &point, const Coords &coords,
                         double s_r) noexcept {
  double dx = coords[0] - point[0];
  double dy = coords[1] - point[1];
  double dz = coords[2] - point[2];
  return (dx * dx + dy * dy + dz * dz) <= s_r ? 1 : 0;
};

template <typename ParticleType> class ParticleList {
private:
  std::vector<ParticleType> particles;

public:
  ParticleList(const std::vector<ParticleType> &particles)
      : particles(particles) {}

  virtual ~ParticleList() = default;
  virtual size_t size() const = 0;
  virtual std::vector<std::vector<double>> getAllCoordinates() const = 0;

  /**
   * @brief Determine the spheres within a radius using a kd tree
   *
   * @param point The reference point as {x, y, z}.
   * @param radius the search radius
   * @return A vector of Sphere indices within the radius.
   */
  std::vector<size_t> collect_kd_indices(const std::vector<double> &point,
                                         double radius) {

    std::vector<size_t> indices = kd_tree.radius_search_indices(point, radius);
    return indices;
  }

  /**
   * @brief Determine the spheres within a radius using a kd tree and return
   * their distance
   *
   * @param point The reference point as {x, y, z}.
   * @param radius the search radius
   * @return A vector of distances for particles within the radius.
   */
  std::vector<double> collect_kd_distances(const std::vector<double> &point,
                                           double radius) {

    std::vector<double> distances =
        kd_tree.radius_search_distances(point, radius);
    return distances;
  }

  /**
   * @brief Collects spheres within a given Verlet radius and returns them as
   * Sphere objects.
   *
   * @param verlet_radius The Verlet radius to consider.
   * @param point The reference point as {x, y, z}.
   * @return A vector of Sphere objects within the Verlet radius.
   */
  std::vector<ParticleType>
  collect_verlet_objects(const std::vector<double> &point,
                         double verlet_radius) {
    std::vector<ParticleType> verlet_particles;
    for (const auto &particle : particles) {
      double r_squared =
          (particle.radius + verlet_radius) * (particle.radius + verlet_radius);

      if (in_sphere(point, particle.coordinates, r_squared)) {
        verlet_particles.push_back(particle);
      }
    }
    return verlet_particles;
  }

  /**
   * @brief Determine the objects within a radius using a kd tree
   * and return indices objects
   * @param point The reference point as {x, y, z}.
   * @param radius the search radius
   * @return A vector of T-type indices within the radius.
   */
  std::vector<ParticleType> collect_kd_objects(const std::vector<double> &point,
                                               double radius) {

    std::vector<ParticleType> local_particles;
    std::vector<size_t> indices = kd_tree.radius_search_indices(point, radius);
    for (const auto &index : indices) {
      local_particles.push_back(particles[index]);
    }
    return local_particles;
  }

  // protected:
  KDTree kd_tree;

  void initializeKDTree(const std::vector<std::vector<double>> &coords) {
    kd_tree.initialize_kd(
        std::make_shared<std::vector<std::vector<double>>>(coords));
  }
};

template <typename ListType>
std::vector<ListType>
convert_to_type(const std::vector<std::vector<double>> &data_in) {
  std::vector<ListType> particle_list;

  for (const auto &data : data_in) {
    // Create a ParticleType from each inner vector and add it to the list
    particle_list.emplace_back(data);
  }

  return particle_list;
}

/**
 * @brief Initializes a SphereList, optionally trims it using KD-tree or
 Verlet,
 * and returns a unique pointer to the SphereList.
 *
 * @param data A vector of data, specific to thew typoe specified
 * @param point The reference point as {x, y, z}.
 * @param radius The search radius for filtering spheres.
 * @param kd_tree Flag to use KD-tree or Verlet algorithm for filtering.
 * @param trim Flag to specify whether to filter the spheres using
 * trim_sphere_list.
 * @return A unique pointer to the initialized (and possibly trimmed)
 * SphereList.
 */
template <typename List, typename ListType>
std::shared_ptr<List>
initialize_list(std::vector<std::vector<double>> &data_in,
                const std::vector<double> &point = {0.0, 0.0, 0.0},
                double radius = 0, bool kd_tree = false, bool trim = false) {

  ;
  std::vector<ListType> data = convert_to_type<ListType>(data_in);
  std::shared_ptr<List> data_list = std::make_shared<List>(data, kd_tree);

  if (trim) {

    if (radius <= 0) {
      throw std::invalid_argument("Radius must be greater than zero.");
    }

    std::vector<ListType> trimmed_data;
    if (kd_tree) {
      trimmed_data = data_list->collect_kd_objects(point, radius);
    } else {
      trimmed_data = data_list->collect_verlet_objects(point, radius);
    }
    // Reinitialize the SphereList with the filtered spheres
    data_list = std::make_shared<List>(trimmed_data, kd_tree);
  }

  return data_list;
}

#endif
