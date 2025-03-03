#ifndef SPHEREPACK_H
#define SPHEREPACK_H

#include "particle_list.hpp"

/**
 * @class Sphere
 * @brief Represents a sphere with position and radius.
 */
class Sphere : public Particle
{
public:
    double radius_squared;

    Sphere(double x, double y, double z, double radius)
        : Particle({ x, y, z }, radius), radius_squared(radius * radius)
    {
    }
    Sphere(std::vector<double> coordinates, double radius)
        : Particle(coordinates, radius), radius_squared(radius * radius)
    {
    }
    Sphere(std::vector<double> sphere_data)
        : Particle(sphere_data, sphere_data[3]),
          radius_squared(sphere_data[3] * sphere_data[3])
    {
    }
};

/**
 * @class SphereList
 * @brief Contains a list of the spheres and tools to operate on those spheres
 */
class SphereList : public ParticleList<Sphere>
{
private:
    std::vector<Sphere> spheres;

    void initializeFromSpheres(std::vector<Sphere> sphere_data, bool build_kd)
    {
        spheres = std::move(sphere_data);

        if (build_kd)
        {
            std::vector<std::vector<double> > coords;
            for (const auto& sphere : spheres)
            {
                coords.push_back({ sphere.coordinates[0],
                                   sphere.coordinates[1],
                                   sphere.coordinates[2] });
            }
            initializeKDTree(coords);
        }
    }

public:
    // Constructor that accepts vector of Sphere objects
    SphereList(std::vector<Sphere> sphere_data, bool build_kd = false)
        : ParticleList<Sphere>(sphere_data)
    {
        initializeFromSpheres(std::move(sphere_data), build_kd);
    }

    /**
     * @brief Determine the largest radius in the list
     * @return the maximum radius in the list of spheres
     * radius.
     */
    double max_radius() const
    {
        return std::max_element(spheres.begin(),
                                spheres.end(),
                                [](const Sphere& a, const Sphere& b)
                                { return a.radius < b.radius; })
            ->radius;
    }
};

#endif