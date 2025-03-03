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

struct Box
{
    double min[3], max[3];
    double length[3];

    // Function to update the lengths of the box
    void box_length()
    {
        length[0] = max[0] - min[0];
        length[1] = max[1] - min[1];
        length[2] = max[2] - min[2];
    }
};

// Particle class
struct Particle
{
    Coords coordinates;
    double radius;
    bool own = false;

    // Constructor to initialize coordinates and radius
    Particle(Coords c = { 0.0, 0.0, 0.0 }, double r = 0.0)
        : coordinates(c), radius(r)
    {
    }
    Particle(double x = 0.0, double y = 0.0, double z = 0.0, double r = 0.0)
        : coordinates({ x, y, z }), radius(r)
    {
    }
};

/**
 * @brief Checks if a point is inside a sphere.
 *
 * @param point The point as {x, y, z}.
 * @param sphere_center The sphere center as {x, y, z}.
 * @param s_r Squared radius of the sphere.
 * @return 1 if the point is inside the sphere, 0 otherwise.
 */
inline uint8_t
in_sphere(const std::vector<double>& point,
          const Coords& coords,
          double s_r) noexcept
{
    double dx = coords[0] - point[0];
    double dy = coords[1] - point[1];
    double dz = coords[2] - point[2];
    return (dx * dx + dy * dy + dz * dz) <= s_r ? 1 : 0;
};

/**
 * @brief Determines if a particle is within a bounding box
 * @param box - the extents of the process
 * @param p - particle with coordinates
 */
bool
inside_box(const Box& box, const Particle& p)
{
    return (p.coordinates[0] >= box.min[0] && p.coordinates[0] <= box.max[0] &&
            p.coordinates[1] >= box.min[1] && p.coordinates[1] <= box.max[1] &&
            p.coordinates[2] >= box.min[2] && p.coordinates[2] <= box.max[2]);
};

/**
 * @brief Determines if a particle intersects bounding box
 * @param box - the extents of the process
 * @param p - particle with coordinates
 */
bool
intersects_box(const Box& box, const Coords& coordinates, const double& radius)
{
    return !(coordinates[0] + radius < box.min[0] ||
             coordinates[0] - radius > box.max[0] ||
             coordinates[1] + radius < box.min[1] ||
             coordinates[1] - radius > box.max[1] ||
             coordinates[2] + radius < box.min[2] ||
             coordinates[2] - radius > box.max[2]);
};

/**
 * @brief Determines if a particle crosses the outer box boundaries.
 * @param particle The particle to check
 * @param outerBox The outer periodic box dimensions
 * @return Array of bools indicating whether the particle crosses each axis
 * boundary.
 */
std::array<bool, 3>
cross_boundary(const Box& box, const Particle& particle)
{
    return {
        particle.coordinates[0] - particle.radius < box.min[0] ||
            particle.coordinates[0] + particle.radius > box.max[0],
        particle.coordinates[1] - particle.radius < box.min[1] ||
            particle.coordinates[1] + particle.radius > box.max[1],
        particle.coordinates[2] - particle.radius < box.min[2] ||
            particle.coordinates[2] + particle.radius > box.max[2],
    };
}

template <typename ParticleType>
class ParticleList
{
protected:
    std::vector<ParticleType> particles;
    KDTree kd_tree;

public:
    ParticleList(const std::vector<ParticleType>& particle_data)
    {
        for (const auto& particle : particle_data)
        {
            particles.emplace_back(particle.coordinates[0],
                                   particle.coordinates[1],
                                   particle.coordinates[2],
                                   particle.radius);
        }
    }

    ~ParticleList() = default;

    size_t size() const
    {
        return particles.size();
    }

    // Overload operator[]
    const ParticleType& operator[](size_t index) const
    {
        if (index >= size())
        {
            throw std::out_of_range("Index out of bounds!");
        }
        return particles[index];
    }

    /**
     * @brief Collect the coordinates of all particles
     *
     * @return A vector of Sphere coordinates
     */
    std::vector<std::vector<double> > getAllCoordinates() const
    {
        std::vector<std::vector<double> > coords;
        for (const auto& particle : particles)
        {
            coords.push_back({ particle.coordinates[0],
                               particle.coordinates[1],
                               particle.coordinates[2] });
        }
        return coords;
    }

    /**
     * @brief Collects particle indices within a given Verlet radius
     *
     * @param verlet_radius The Verlet radius to consider.
     * @param point The reference point as {x, y, z}.
     * @return A vector of indices to particles within the Verlet radius.
     */
    std::vector<size_t> collect_verlet_indices(const std::vector<double>& point,
                                               double verlet_radius)
    {
        std::vector<size_t> verlet_indices;
        for (size_t i = 0; i < particles.size(); ++i)
        {
            const auto& particle = particles[i];
            double r_squared = (particle.radius + verlet_radius) *
                               (particle.radius + verlet_radius);

            if (in_sphere(point, particle.coordinates, r_squared))
            {
                verlet_indices.push_back(i);
            }
        }
        return verlet_indices;
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
    collect_verlet_objects(const std::vector<double>& point,
                           double verlet_radius)
    {
        std::vector<ParticleType> verlet_particles;
        for (const auto& particle : particles)
        {
            double r_squared = (particle.radius + verlet_radius) *
                               (particle.radius + verlet_radius);

            if (in_sphere(point, particle.coordinates, r_squared))
            {
                verlet_particles.push_back(particle);
            }
        }
        return verlet_particles;
    }

    void initializeKDTree(const std::vector<std::vector<double> >& coords)
    {
        kd_tree.initialize_kd(
            std::make_shared<std::vector<std::vector<double> > >(coords));
    }

    /**
     * @brief Determine the spheres within a radius using a kd tree
     *
     * @param point The reference point as {x, y, z}.
     * @param radius the search radius
     * @return A vector of Sphere indices within the radius.
     */
    std::vector<size_t> collect_kd_indices(const std::vector<double>& point,
                                           double radius)
    {
        std::vector<size_t> indices =
            kd_tree.radius_search_indices(point, radius);
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
    std::vector<double> collect_kd_distances(const std::vector<double>& point,
                                             double radius,
                                             bool return_square = true)
    {
        std::vector<double> distances =
            kd_tree.radius_search_distances(point, radius, return_square);
        return distances;
    }

    /**
     * @brief Return a vector of the particles with {x,y,z,radius,own}
     * @param own: Return own status
     * @return A vector of  particles with {x,y,z,radius,own}
     */
    std::vector<std::vector<double> > return_particles(bool own = false)
    {
        std::vector<std::vector<double> > info;
        info.reserve(particles.size());
        for (const auto& particle : particles)
        {
            std::vector<double> _info = { particle.coordinates[0],
                                          particle.coordinates[1],
                                          particle.coordinates[2],
                                          particle.radius };

            if (own) _info.push_back(static_cast<double>(particle.own));

            info.emplace_back(std::move(_info)); // Move to avoid extra copies
        }

        return info;
    }

    /**
     * @brief Determine the objects within a radius using a kd tree
     * and return indices objects
     * @param point The reference point as {x, y, z}.
     * @param radius the search radius
     * @return A vector of T-type indices within the radius.
     */
    std::vector<ParticleType>
    collect_kd_objects(const std::vector<double>& point, double radius)
    {
        std::vector<ParticleType> local_particles;
        std::vector<size_t> indices =
            kd_tree.radius_search_indices(point, radius);
        for (const auto& index : indices)
        {
            local_particles.push_back(particles[index]);
        }
        return local_particles;
    }

    /**
     * @brief Determines if particles are within a bounding box
     * @param box - the extents of the process
     */
    void particles_in_box(const Box& box)
    {
        for (auto& p : particles)
        {
            p.own = inside_box(box, p);
        }
    }

    /**
     * @brief Provide access to the particles
     */
    // Provide a getter for atoms if you want to access it directly
    const std::vector<ParticleType>& getParticles() const
    {
        return particles;
    }

    /**
     * @brief Applies periodic boundary conditions to spheres and retains only
     * those that intersect the inner box.
     * @param domain Dimensions of domain via Box
     * @param subdomain Dimensions of the subdomain vix Box
     */
    void add_periodic_particles(const Box& domain, const Box& subdomain)
    {
        std::vector<ParticleType> new_particles;
        for (const auto& particle : particles)
        {
            auto crosses = cross_boundary(domain, particle);

            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        if (dx == 0 && dy == 0 && dz == 0)
                        {
                            if (intersects_box(subdomain,
                                               particle.coordinates,
                                               particle.radius))
                            {
                                new_particles.push_back(particle);
                            }
                            continue; // Skip original sphere
                        }

                        if ((dx != 0 && crosses[0]) ||
                            (dy != 0 && crosses[1]) || (dz != 0 && crosses[2]))
                        {
                            Coords periodic_particle = {
                                particle.coordinates[0] + dx * domain.length[0],
                                particle.coordinates[1] + dy * domain.length[1],
                                particle.coordinates[2] + dz * domain.length[2],
                            };
                            if (intersects_box(subdomain,
                                               periodic_particle,
                                               particle.radius))
                            {
                                new_particles.push_back(
                                    { periodic_particle, particle.radius });
                            }
                        }
                    }
                }
            }
        }

        particles = std::move(new_particles);
    }

    /**
     * @brief Removes particles that do not intersect specified box
     * @param subdomain Dimensions of the subdomain vix Box
     */
    void trim_particles(const Box& subdomain)
    {
        std::vector<ParticleType> new_particles;
        for (const auto& particle : particles)
        {
            if (intersects_box(
                    subdomain, particle.coordinates, particle.radius))
            {
                new_particles.push_back(particle);
            }
        }
        particles = std::move(new_particles);
    }

}; // end class

template <typename ListType>
std::vector<ListType>
convert_to_type(const std::vector<std::vector<double> >& data_in)
{
    std::vector<ListType> particle_list;

    for (const auto& data : data_in)
    {
        particle_list.emplace_back(data);
    }

    return particle_list;
}

/**
 * @brief Initializes a SphereList, optionally trims it using KD-tree or
 * Verlet, and returns a unique pointer to the SphereList.
 *
 * @param data A vector of data, specific to thew typoe specified
 * @param point The reference point as {x, y, z}.
 * @param radius The search radius for filtering spheres.
 * @param kd_tree Flag to use KD-tree or Verlet algorithm for filtering.
 * @param add_periodic Flag to specify whether periodic particles should be
 * added
 * @return A unique pointer to the initialized (and possibly trimmed)
 * SphereList.
 */
template <typename List, typename ListType>
std::shared_ptr<List>
initialize_list(std::vector<std::vector<double> >& data_in,
                const std::vector<std::vector<double> >& domain,
                const std::vector<std::vector<double> >& subdomain,
                bool add_periodic = false)
{
    std::vector<ListType> data = convert_to_type<ListType>(data_in);

    Box domain_box = { { domain[0][0], domain[1][0], domain[2][0] },
                       { domain[0][1], domain[1][1], domain[2][1] } };

    domain_box.box_length();

    Box subdomain_box = {
        { subdomain[0][0], subdomain[1][0], subdomain[2][0] },
        { subdomain[0][1], subdomain[1][1], subdomain[2][1] }
    };

    std::shared_ptr<List> data_list = std::make_shared<List>(data);

    if (add_periodic)
    {
        data_list->add_periodic_particles(domain_box, subdomain_box);
    }
    else
    {
        data_list->trim_particles(subdomain_box);
    }

    if (data_list->size() < 1)
    {
        throw std::invalid_argument("List has not entries after trimming.");
    }

    return data_list;
}

template <typename List>
std::vector<std::vector<double> >
return_particles(std::shared_ptr<List> particle_list, bool return_own = false)
{
    return particle_list->return_particles(return_own);
}

template <typename List>
void
set_own_particles(std::shared_ptr<List> particle_list,
                  const std::vector<std::vector<double> >& subdomain)
{
    Box box = { { subdomain[0][0], subdomain[1][0], subdomain[2][0] },
                { subdomain[0][1], subdomain[1][1], subdomain[2][1] } };
    particle_list->particles_in_box(box);
}

#endif
