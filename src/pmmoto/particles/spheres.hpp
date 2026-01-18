#ifndef SPHERES_H
#define SPHERES_H

#include "particle_list.hpp"
#include "shape.hpp"

/**
 * @class Sphere
 * @brief Represents a sphere with position and radius.
 */
class Sphere : public Particle, public Shape
{
public:
    double radius;
    double radius_squared;
    double mass = 0;

    Sphere(double x, double y, double z, double radius)
        : Particle(x, y, z), radius(radius), radius_squared(radius * radius)
    {
    }
    Sphere(std::vector<double> coordinates, double radius)
        : Particle(coordinates), radius(radius), radius_squared(radius * radius)
    {
    }
    Sphere(std::vector<double> sphere_data)
        : Particle(sphere_data),
          radius(sphere_data[3]),
          radius_squared(sphere_data[3] * sphere_data[3])
    {
    }

    /**
     * @brief Checks if a point is inside a sphere.
     *
     * @param point The point as {x, y, z}.
     * @param sphere_center The sphere center as {x, y, z}.
     * @param s_r Squared radius of the sphere.
     * @return 1 if the point is inside the sphere, 0 otherwise.
     */
    inline uint8_t
    contains(const std::vector<double>& voxel) const noexcept override
    {
        double dx = coordinates[0] - voxel[0];
        double dy = coordinates[1] - voxel[1];
        double dz = coordinates[2] - voxel[2];
        return (dx * dx + dy * dy + dz * dz) <= radius_squared ? 1 : 0;
    };

    /**
     * @brief Determines if a particle intersects bounding box
     * @param box - the extents of the process
     * @param p - particle with coordinates
     */
    static inline bool
    intersects_box(const Coords& coordinates, double radius, const Box& box)
    {
        return !(coordinates[0] + radius < box.min[0] ||
                 coordinates[0] - radius > box.max[0] ||
                 coordinates[1] + radius < box.min[1] ||
                 coordinates[1] - radius > box.max[1] ||
                 coordinates[2] + radius < box.min[2] ||
                 coordinates[2] - radius > box.max[2]);
    }

    /**
     * @brief Determines if a sphere crosses the boundaries of box
     * @param box Box stuct
     * @return Array of bools indicating whether the particle crosses each axis
     * boundary.
     */
    inline std::array<bool, 3> cross_boundary(const Box& box) const
    {
        std::array<bool, 3> crosses{};

        for (std::size_t d = 0; d < 3; ++d)
        {
            const bool crosses_lower = coordinates[d] - radius < box.min[d];
            const bool crosses_upper = coordinates[d] + radius > box.max[d];
            crosses[d] = crosses_lower || crosses_upper;
        }

        return crosses;
    }

    /**
     * @brief Determines whether a sphere extends beyond the box extents.
     *
     * This function performs a purely geometric check and does NOT consider
     * periodicity or boundary conditions.
     *
     * @param box Axis-aligned bounding box
     * @return Array indicating extension beyond box in each dimension
     */
    inline std::array<bool, 3> extends_outside_box(const Box& box) const
    {
        std::array<bool, 3> outside{};

        for (std::size_t d = 0; d < 3; ++d)
        {
            outside[d] = (coordinates[d] - radius < box.min[d]) ||
                         (coordinates[d] + radius > box.max[d]);
        }

        return outside;
    }

    /**
     * @brief Print sphere coordinates and radius
     */
    void print() const
    {
        std::cout << coordinates[0] << " " << coordinates[1] << " "
                  << coordinates[2] << " " << radius << std::endl;
    }
};

/**
 * @class SphereList
 * @brief Contains a list of the spheres and tools to operate on those spheres
 */
class SphereList : public ShapeList
{
protected:
    std::vector<Sphere> spheres;
    ParticleList particle_list;
    size_t own_count{ 0 };

public:
    SphereList(const std::vector<std::vector<double> >& coordinates,
               const std::vector<double>& radius)
        : particle_list(coordinates)
    {
        const auto& particles = particle_list.getParticles();

        // Assuming the number of particles matches the number of radii
        for (size_t i = 0; i < particles.size(); ++i)
        {
            spheres.push_back(Sphere(particles[i].coordinates, radius[i]));
        }
    }

    SphereList(const std::vector<std::vector<double> >& coordinates,
               const double& radius)
        : particle_list(coordinates)
    {
        const auto& particles = particle_list.getParticles();

        // Assuming the number of particles matches the number of radii
        for (size_t i = 0; i < particles.size(); ++i)
        {
            spheres.push_back(Sphere(particles[i].coordinates, radius));
        }
    }

    std::shared_ptr<Shape> get(size_t index) const override
    {
        return std::make_shared<Sphere>(spheres[index]);
    }

    /**
     * @brief Get the number of spheres
     */
    size_t size() const
    {
        return spheres.size();
    }

    /**
     * @brief Set masses
     */
    void set_masses(const std::vector<double>& masses)
    {
        for (size_t i = 0; i < spheres.size(); ++i)
        {
            spheres[i].mass = masses[i];
        }
    }

    /**
     * @brief Get masses
     */
    std::vector<double> get_masses() const
    {
        std::vector<double> masses;
        masses.reserve(spheres.size());
        for (size_t i = 0; i < spheres.size(); ++i)
        {
            masses.emplace_back(spheres[i].mass);
        }

        return masses;
    }

    /**
     * @brief Build a kd-tree
     */
    void build_KDtree()
    {
        particle_list.initializeKDTree();
    }

    // Add operator[] to access individual spheres
    Sphere& operator[](size_t index)
    {
        if (index >= spheres.size())
        {
            throw std::out_of_range("Sphere index out of range");
        }
        return spheres[index];
    }

    const Sphere& operator[](size_t index) const
    {
        if (index >= spheres.size())
        {
            throw std::out_of_range("Sphere index out of range");
        }
        return spheres[index];
    }

    /**
     * @brief Determine the maximum sphere radius
     */
    double max_radius()
    {
        if (spheres.empty())
        {
            return 0.0;
        }

        return std::max_element(spheres.begin(),
                                spheres.end(),
                                [](const Sphere& a, const Sphere& b)
                                { return a.radius < b.radius; })
            ->radius;
    }

    /**
     * @brief Provide access to the particles
     * @note This class can update the sphereList so need to grab actual
     * coordinates.
     */
    std::vector<std::vector<double> > get_coordinates() const
    {
        return particle_list.get_coordinates();
    }

    std::vector<std::vector<double> > return_spheres(bool return_own = false)
    {
        std::vector<std::vector<double> > info;
        info.reserve(spheres.size());
        for (const auto& sphere : spheres)
        {
            std::vector<double> _info = { sphere.coordinates[0],
                                          sphere.coordinates[1],
                                          sphere.coordinates[2],
                                          sphere.radius };

            if (return_own) _info.push_back(static_cast<double>(sphere.own));

            info.emplace_back(std::move(_info)); // Move to avoid extra
                                                 // copies
        }

        return info;
    }

    /**
     * @brief Determines if particles are within a bounding box
     * @param box - the extents of the process
     */
    void own_spheres(const Box& box)
    {
        own_count = 0;
        for (auto& sphere : spheres)
        {
            sphere.own = sphere.inside_box(box);
            if (sphere.own) own_count++;
        }
    }

    /**
     * @brief Get the number of owned spheres
     * @return Number of spheres owned by this process
     */
    size_t get_own_count() const
    {
        return own_count;
    }

    /**
     * @brief Finds indices of spheres that intersect with the specified
     box
     * @param subdomain Dimensions of the subdomain via Box
     * @return Vector of indices of intersecting spheres
     */
    std::vector<size_t> find_intersecting_indices(const Box& box) const override
    {
        std::vector<size_t> indices;

        for (size_t i = 0; i < spheres.size(); ++i)
        {
            const Sphere& sphere = spheres[i];
            if (sphere.intersects_box(sphere.coordinates, sphere.radius, box))
            {
                indices.push_back(i);
            }
        }
        return indices;
    }

    /**
     * @brief Removes spheres that do not intersect specified box
     * @param subdomain Dimensions of the subdomain vix Box
     */
    void trim_spheres_intersecting(const Box& subdomain)
    {
        std::vector<Coords> new_coords;
        std::vector<Sphere> new_spheres;

        for (auto& sphere : spheres)
        {
            if (sphere.intersects_box(
                    sphere.coordinates, sphere.radius, subdomain))
            {
                new_coords.push_back(sphere.coordinates);
                new_spheres.push_back(sphere);
            }
        }

        particle_list.updateParticles(new_coords);
        spheres = std::move(new_spheres);
    }

    /**
     * @brief Removes spheres that are not within specified box
     * @param subdomain Dimensions of the subdomain vix Box
     */
    void trim_spheres_within(const Box& box)
    {
        std::vector<Coords> new_coords;
        std::vector<Sphere> new_spheres;

        for (auto& sphere : spheres)
        {
            if (sphere.inside_box(box))
            {
                new_coords.push_back(sphere.coordinates);
                new_spheres.push_back(sphere);
            }
        }

        particle_list.updateParticles(new_coords);
        spheres = std::move(new_spheres);
    }

    /**
     * @brief Applies periodic boundary conditions to spheres and retains
     only
     * those that intersect the inner box.
     * @param domain Dimensions of domain via Box with periodic boundary info
     * @param subdomain Dimensions of the subdomain vix Box
     */
    void add_periodic_spheres(const Box& domain, const Box& subdomain)
    {
        std::vector<Sphere> new_spheres;
        std::vector<Coords> new_coords;

        static constexpr std::array<int, 3> shifts = { -1, 0, 1 };

        for (const auto& sphere : spheres)
        {
            const auto extends = sphere.extends_outside_box(domain);

            for (int dx : shifts)
            {
                if (dx != 0 && !(extends[0] && domain.periodic[0])) continue;

                for (int dy : shifts)
                {
                    if (dy != 0 && !(extends[1] && domain.periodic[1]))
                        continue;

                    for (int dz : shifts)
                    {
                        if (dz != 0 && !(extends[2] && domain.periodic[2]))
                            continue;

                        Coords image = sphere.coordinates;

                        image[0] += dx * domain.length[0];
                        image[1] += dy * domain.length[1];
                        image[2] += dz * domain.length[2];

                        if (Sphere::intersects_box(
                                image, sphere.radius, subdomain))
                        {
                            new_coords.push_back(image);
                            new_spheres.push_back({ image, sphere.radius });
                        }
                    }
                }
            }
        }

        particle_list.updateParticles(new_coords);
        spheres = std::move(new_spheres);
    }

    /**
     * @brief Determine the spheres within a radius using a kd tree and
     * return their distance
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
            particle_list.collect_kd_distances(point, radius, return_square);
        return distances;
    }

    /**
     * @brief Determine the indices of spheres within a radius using a kd
     * tree
     * @param point The reference point as {x, y, z}.
     * @param radius the search radius
     * @return A vector of distances for particles within the radius.
     */
    std::vector<size_t> collect_kd_indices(const std::vector<double>& point,
                                           double radius,
                                           bool return_square = true)
    {
        std::vector<size_t> distances =
            particle_list.collect_kd_indices(point, radius);
        return distances;
    }
};

#endif