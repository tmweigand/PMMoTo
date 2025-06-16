#ifndef SHAPE_H
#define SHAPE_H

class Shape
{
public:
    virtual ~Shape() = default;
    virtual inline uint8_t contains(const std::vector<double>& voxel) const = 0;
};

// Forward declare Box
struct Box;

// Abstract base for shape lists
class ShapeList
{
public:
    virtual ~ShapeList() = default;

    // Virtual method to override for shape-specific intersection logic
    virtual std::vector<size_t>
    find_intersecting_indices(const Box& box) const = 0;

    virtual std::shared_ptr<Shape> get(size_t index) const = 0;
};

#endif