#ifndef SPHEREPACK_H
#define SPHEREPACK_H

#include "particle_list.hpp"

/**
 * @class Sphere
 * @brief Represents a sphere with position and radius.
 */
class Sphere : public Particle {
public:
  double radius_squared;

  Sphere(double x, double y, double z, double radius)
      : Particle({x, y, z}, radius), radius_squared(radius * radius) {}
  Sphere(std::vector<double> coordinates, double radius)
      : Particle(coordinates, radius), radius_squared(radius * radius) {}
  Sphere(std::vector<double> sphere_data)
      : Particle(sphere_data, sphere_data[3]),
        radius_squared(sphere_data[3] * sphere_data[3]) {}
};

/**
 * @class SphereList
 * @brief Contains a list of the spheres and tools to operate on those spheres
 */
class SphereList : public ParticleList<Sphere> {

private:
  std::vector<Sphere> spheres;
  KDTree kd_tree;

  void initializeFromSpheres(std::vector<Sphere> sphere_data, bool build_kd) {
    spheres = std::move(sphere_data);

    if (build_kd) {
      std::vector<std::vector<double>> coords;
      for (const auto &sphere : spheres) {
        coords.push_back({sphere.coordinates[0], sphere.coordinates[1],
                          sphere.coordinates[2]});
      }
      initializeKDTree(coords);
    }
  }

public:
  // Constructor that accepts vector of Sphere objects
  SphereList(std::vector<Sphere> sphere_data, bool build_kd = false)
      : ParticleList<Sphere>(sphere_data) {
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
      coords.push_back({sphere.coordinates[0], sphere.coordinates[1],
                        sphere.coordinates[2]});
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

      if (in_sphere(point, sphere.coordinates, r_squared)) {
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

      if (in_sphere(point, sphere.coordinates, r_squared)) {
        verlet_spheres.push_back(sphere);
      }
    }
    return verlet_spheres;
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

  void initialize_kd() {
    std::vector<std::vector<double>> coords;
    for (const auto &sphere : spheres) {
      coords.push_back({sphere.coordinates[0], sphere.coordinates[1],
                        sphere.coordinates[2]});
    }
    std::cout << "Coords length: " << coords.size() << "\n";
    initializeKDTree(coords);
  }
};

// /**
//  * @brief Initializes a SphereList, optionally trims it using KD-tree or
//  Verlet,
//  * and returns a shared pointer to the SphereList.
//  *
//  * @param sphere_data A vector of spheres, each represented as {x, y, z, r}.
//  * @param point The reference point as {x, y, z}.
//  * @param radius The search radius for filtering spheres.
//  * @param kd_tree Flag to use KD-tree or Verlet algorithm for filtering.
//  * @param trim Flag to specify whether to filter the spheres using
//  * trim_sphere_list.
//  * @return A shared pointer to the initialized (and possibly trimmed)
//  * SphereList.
//  */
// std::shared_ptr<SphereList>
// initialize_sphere_list(std::vector<std::vector<double>> &sphere_data,
//                        const std::vector<double> &point = {0.0, 0.0, 0.0},
//                        double radius = 0, bool kd_tree = false,
//                        bool trim = false) {

//   auto sphere_list = std::make_shared<SphereList>(sphere_data, kd_tree);

//   if (trim) {

//     if (radius <= 0) {
//       throw std::invalid_argument("Radius must be greater than zero.");
//     }

//     std::vector<Sphere> trimmed_spheres;
//     if (kd_tree) {
//       trimmed_spheres = sphere_list->collect_kd_spheres_objects(point,
//       radius);
//     } else {
//       trimmed_spheres =
//           sphere_list->collect_verlet_sphere_objects(radius, point);
//     }
//     // Reinitialize the SphereList with the filtered spheres
//     sphere_list = std::make_shared<SphereList>(trimmed_spheres, kd_tree);
//   }

//   return sphere_list;
// }

// /**
//  * @brief Initializes a SphereList, filters spheres using KD-tree, and
//  * reinitializes with the result.
//  *
//  * @param sphere_data A vector of spheres, each represented as {x, y, z, r}.
//  * @param point The reference point as {x, y, z}.
//  * @param radius The search radius for filtering spheres.
//  * @return A new SphereList containing only the spheres within the given
//  radius.
//  */
// std::shared_ptr<SphereList>
// trim_sphere_list(std::vector<std::vector<double>> &sphere_data_in,
//                  const std::vector<double> &point, double radius,
//                  bool kd_tree = false) {
//   // Step 1: Initialize SphereList with the original sphere data and build
//   // KD-tree
//   auto sphere_data = convert_to_type<Sphere>(sphere_data_in);
//   SphereList sphere_list(sphere_data, kd_tree);

//   std::vector<Sphere> filtered_spheres;
//   if (kd_tree) {
//     filtered_spheres = sphere_list.collect_kd_spheres_objects(point, radius);
//   } else {
//     filtered_spheres = sphere_list.collect_verlet_sphere_objects(radius,
//     point);
//   }

//   return std::make_shared<SphereList>(filtered_spheres, kd_tree);
// }

// /**
//  * @brief Initializes a SphereList, filters spheres using KD-tree, and
//  * reinitializes with the result.
//  *
//  * @param sphere_data A vector of spheres, each represented as {x, y, z, r}.
//  * @param point The reference point as {x, y, z}.
//  * @param radius The search radius for filtering spheres.
//  * @return A new SphereList containing only the spheres within the given
//  radius.
//  */
// std::vector<Sphere>
// trim_sphere_list_spheres(std::vector<std::vector<double>> &sphere_data_in,
//                          const std::vector<double> &point, double radius,
//                          bool kd_tree = false) {
//   // Step 1: Initialize SphereList with the original sphere data and build
//   // KD-tree
//   auto sphere_data = convert_to_type<Sphere>(sphere_data_in);
//   SphereList sphere_list(sphere_data, kd_tree);

//   std::vector<Sphere> filtered_spheres;
//   if (kd_tree) {
//     filtered_spheres = sphere_list.collect_kd_spheres_objects(point, radius);
//   } else {
//     filtered_spheres = sphere_list.collect_verlet_sphere_objects(radius,
//     point);
//   }

//   return filtered_spheres;
// }

#endif