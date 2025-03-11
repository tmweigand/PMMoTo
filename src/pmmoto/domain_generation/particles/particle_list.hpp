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
    double min[3], max[3], length[3];

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
    // double radius;
    bool own = false;

    // Constructor to initialize coordinates
    Particle(Coords c = { 0.0, 0.0, 0.0 }) : coordinates(c)
    {
    }
    Particle(double x = 0.0, double y = 0.0, double z = 0.0)
        : coordinates({ x, y, z })
    {
    }

    void print()
    {
        std::cout << coordinates[0] << " " << coordinates[1] << " "
                  << coordinates[2] << std::endl;
    }

    /**
     * @brief Determines if a particle is within a bounding box
     * @param box - the extents of the process
     */
    void inside_box(const Box& box)
    {
        own = (coordinates[0] >= box.min[0] && coordinates[0] <= box.max[0] &&
               coordinates[1] >= box.min[1] && coordinates[1] <= box.max[1] &&
               coordinates[2] >= box.min[2] && coordinates[2] <= box.max[2]);
    };
};

class ParticleList
{
private:
    std::vector<Particle> particles;
    std::shared_ptr<KDTree> kd_tree;

public:
    // Update constructors to initialize kd_tree
    ParticleList(const std::vector<Particle>& particle_data)
        : particles(particle_data), kd_tree(std::make_shared<KDTree>())
    {
    }

    ParticleList(const std::vector<Coords>& particle_data)
        : kd_tree(std::make_shared<KDTree>())
    {
        for (const auto& coords : particle_data)
        {
            particles.push_back(Particle{ coords });
        }
    }

    // Move constructor with noexcept
    ParticleList(ParticleList&& other) noexcept = default;

    // Move assignment operator with noexcept
    ParticleList& operator=(ParticleList&& other) noexcept = default;

    size_t size() const
    {
        return particles.size();
    }

    /**
     * @brief Method to update particles
     */
    void updateParticles(const std::vector<Coords>& new_coords)
    {
        particles.clear();
        particles.reserve(new_coords.size());

        for (const auto& coords : new_coords)
        {
            particles.emplace_back(Particle{ coords });
        }
    }

    // Overload operator[]
    const Particle& operator[](size_t index) const
    {
        if (index >= size())
        {
            throw std::out_of_range("Index out of bounds!");
        }

        return particles[index];
    }

    /**
     * @brief Collect the coordinates of all particles
     * @return A vector of coordinates
     */
    std::vector<std::vector<double> > get_coordinates() const
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
     * @brief Initialize kd tree
     */
    void initializeKDTree()
    {
        if (!kd_tree)
        {
            kd_tree = std::make_shared<KDTree>();
        }
        auto coords = std::make_shared<std::vector<std::vector<double> > >(
            get_coordinates());
        kd_tree->initialize_kd(coords);
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
        if (!kd_tree)
        {
            throw std::runtime_error("KD-tree not initialized");
        }
        return kd_tree->radius_search_indices(point, radius);
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
            kd_tree->radius_search_distances(point, radius, return_square);
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
                                          particle.coordinates[2] };

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
    std::vector<Coords> collect_kd_objects(const std::vector<double>& point,
                                           double radius)
    {
        std::vector<Coords> local_particles;
        std::vector<size_t> indices =
            kd_tree->radius_search_indices(point, radius);
        for (const auto& index : indices)
        {
            local_particles.push_back(particles[index].coordinates);
        }
        return local_particles;
    }

    /**
     * @brief Determines if particles are within a bounding box
     * @param box - the extents of the process
     */
    void particles_in_box(const Box& box)
    {
        for (auto& particle : particles)
        {
            particle.inside_box(box);
        }
    }

    /**
     * @brief Provide access to the particles
     */
    // Provide a getter for atoms if you want to access it directly
    const std::vector<Particle>& getParticles() const
    {
        return particles;
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
 * @brief Initializes a List of particles.
 *
 * @param data A vector of data, specific to thew type specified
 * @return A unique pointer to the initialized (and possibly trimmed)
 * ParticleList.
 */
std::shared_ptr<ParticleList>
initialize_particles(std::vector<std::vector<double> >& data_in)
{
    std::vector<Particle> data = convert_to_type<Particle>(data_in);

    std::shared_ptr<ParticleList> data_list =
        std::make_shared<ParticleList>(data);

    return data_list;
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
