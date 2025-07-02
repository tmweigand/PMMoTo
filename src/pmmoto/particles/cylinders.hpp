#ifndef CYLINDERS_H
#define CYLINDERS_H

#include "particle_list.hpp"
#include "shape.hpp"

/**
 * @class Cylinder
 * @brief Represents a cylinder withtwo coordinates and a radius.
 */
class Cylinder : public Shape
{
private:
    std::vector<double> point_1, point_2;
    std::vector<double> v; // Normalized axis direction
    double radius;
    double h;

    static double dot(const std::vector<double>& a,
                      const std::vector<double>& b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

public:
    Cylinder(std::vector<double> point_1,
             std::vector<double> point_2,
             double radius)
        : point_1(point_1), point_2(point_2), radius(radius)
    {
        // Compute axis vector and normalize it
        v = { point_2[0] - point_1[0],
              point_2[1] - point_1[1],
              point_2[2] - point_1[2] };
        h = std::sqrt(dot(v, v));
        for (int i = 0; i < 3; ++i) v[i] /= h; // Normalize axis direction
    }

    /**
     * @brief Checks if a point is inside a cylinder.
     *
     * @param point The point as {x, y, z}.
     * @return 1 if the point is inside the sphere, 0 otherwise.
     */
    inline uint8_t
    contains(const std::vector<double>& voxel) const noexcept override
    {
        std::vector<double> w = { voxel[0] - point_1[0],
                                  voxel[1] - point_1[1],
                                  voxel[2] - point_1[2] };

        // Project w onto axis direction
        double t = dot(w, v);

        // Check if projection lies within [0, h]
        if (t < 0.0 || t > h) return false;

        // Compute squared distance to axis
        double w_len2 = dot(w, w);
        double d2 = w_len2 - t * t;

        return d2 <= radius * radius;
    };
};

/**
 * @class CylinderList
 * @brief Contains a list of the cylinders and tools to operate on those
 * cylinders
 */
class CylinderList : public ShapeList
{
protected:
    std::vector<Cylinder> cylinders;
    size_t own_count{ 0 };

public:
    CylinderList(const std::vector<std::vector<double> >& point_1,
                 const std::vector<std::vector<double> >& point_2,
                 const std::vector<double>& radius)
    {
        // Assuming the number of particles matches the number of radii
        for (size_t i = 0; i < radius.size(); ++i)
        {
            cylinders.push_back(Cylinder(point_1[i], point_2[i], radius[i]));
        }
    }

    std::shared_ptr<Shape> get(size_t index) const override
    {
        return std::make_shared<Cylinder>(cylinders[index]);
    }

    /**
     * @brief Finds indices of cylinders that intersect with the specified
     box
     * @param subdomain Dimensions of the subdomain via Box
     * @return Vector of indices of intersecting spheres
     * @note This currently returns all cylinders.
     */
    std::vector<size_t> find_intersecting_indices(const Box& box) const override
    {
        std::vector<size_t> indices;

        for (size_t i = 0; i < cylinders.size(); ++i)
        {
            indices.push_back(i);
        }
        return indices;
    }

    // Add operator[] to access individual spheres
    Cylinder& operator[](size_t index)
    {
        if (index >= cylinders.size())
        {
            throw std::out_of_range("Cylinder index out of range");
        }
        return cylinders[index];
    }
};
#endif