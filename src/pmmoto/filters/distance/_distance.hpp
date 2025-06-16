#ifndef DISTANCE_H
#define DISTANCE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#define sq(x) (static_cast<float>(x) * static_cast<float>(x))

struct Hull
{
    int vertex;
    float height;
    float range;
};

inline void
to_finite(float* f, const size_t voxels)
{
    for (size_t i = 0; i < voxels; i++)
    {
        if (f[i] == INFINITY)
        {
            f[i] = std::numeric_limits<float>::max() - 1;
        }
    }
}

inline void
to_infinite(float* f, const size_t voxels)
{
    for (size_t i = 0; i < voxels; i++)
    {
        if (f[i] >= std::numeric_limits<float>::max() - 1)
        {
            f[i] = INFINITY;
        }
    }
}

/**
 * @brief Calculates the point of intersection between two parabolas.
 *
 * This function computes the horizontal coordinate of the intersection point
 * between two parabolas defined by their focal distances and vertex
 * coordinates.
 *
 * @param f_r The focal distance of the first parabola.
 * @param r The vertex coordinate of the first parabola.
 * @param f_q The focal distance of the second parabola.
 * @param q The vertex coordinate of the second parabola.
 * @param anisotropy_factor A factor to account for anisotropic scaling.
 * @return The horizontal coordinate of the intersection point.
 */
inline float
intersect_parabolas(const float f_r,
                    const float r,
                    const float f_q,
                    const float q,
                    const float anistropy_factor)
{
    float factor1, factor2, s;
    factor1 = (r - q) * anistropy_factor;
    factor2 = r + q;
    s = (f_r - f_q + factor1 * factor2) / (2.0 * factor1);
    return s;
}

/**
 * Function to update the convex hull structure with a new point.
 *
 * @param k              Current index of the leading parabola
 * @param i              Current index in the main loop.
 * @param ff             Function value.
 * @param hull_vertices  Vector storing the indices of the convex hull
 *                       vertices.
 * @param ranges         Vector holding the ranges for each hull segment.
 * @param anisotropy_factor A factor to account for anisotropic scaling.
 *
 * This function computes the intersection of parabolas, updates the hull
 * structure by maintaining the convexity property, and adjusts the range
 * boundaries accordingly.
 */
void
update_hull(int& k,
            int i,
            float ff,
            std::vector<int>& hull_vertices,
            std::vector<float>& hull_height,
            std::vector<float>& ranges,
            float anisotropy_factor)
{
    float s = intersect_parabolas(
        ff, i, hull_height[k], hull_vertices[k], anisotropy_factor);
    while (k > 0 && s <= ranges[k])
    {
        k--;
        s = intersect_parabolas(
            ff, i, hull_height[k], hull_vertices[k], anisotropy_factor);
    }

    k++;
    hull_vertices[k] = i;
    hull_height[k] = ff;
    ranges[k] = s;
    ranges[k + 1] = INFINITY;
}

/**
 * @brief Computes the squared Euclidean distance transform (EDT) for a 1D
 * array with multiple segments.
 *
 * This function calculates the squared Euclidean distance transform for a
 * segmented 1D array. It processes segments independently, applying specified
 * anisotropy and correction factors.
 *
 * @tparam T Type of the segment identifiers.
 * @param[in] img Pointer to the array of segment identifiers. A value of 0
 * indicates no segment.
 * @param[out] d Pointer to the distance array, which will be updated with
 * squared distances.
 * @param[in] n Number of elements in the array.
 * @param[in] stride Stride value used for indexing into the arrays (useful
 * for multi-dimensional arrays).
 * @param[in] anistropy The anisotropy factor applied to distances between
 * elements.
 * @param[in] lower_corrector Distance to lower solid. Defaults to `INFINITY`.
 * @param[in] upper_corrector Distance to upper solid. Defaults to `INFINITY`.
 */
template <typename T>
void
squared_edt_1d(T* img,
               float* d,
               const int n,
               const long int stride,
               const float resolution,
               const float lower_corrector = INFINITY,
               const float upper_corrector = INFINITY)
{
    long int i;
    T working_segid = img[0];
    d[0] = working_segid == 0 ? 0 : lower_corrector * resolution;

    for (i = stride; i < n * stride; i += stride)
    {
        if (img[i] == 0)
        {
            d[i] = 0.0;
        }
        else if (img[i] == working_segid)
        {
            d[i] = d[i - stride] + resolution;
        }
        else
        {
            d[i] = resolution;
            d[i - stride] =
                static_cast<float>(img[i - stride] != 0) * resolution;
            working_segid = img[i];
        }
    }

    if (d[(n - 1) * stride] > upper_corrector * resolution)
    {
        d[(n - 1) * stride] = upper_corrector * resolution;
    }

    for (i = (n - 2) * stride; i >= 0; i -= stride)
    {
        d[i] = std::fminf(d[i], d[i + stride] + resolution);
    }

    for (i = 0; i < n * stride; i += stride)
    {
        d[i] *= d[i];
    }
}

void
squared_edt_1d_parabolic(float* img,
                         const int n,
                         const float resolution,
                         const long int stride,
                         std::vector<Hull> lower_hull,
                         std::vector<Hull> upper_hull)
{
    if (n == 0)
    {
        return;
    }

    const float anisotropy_factor = resolution * resolution;

    std::vector<float> ff(n, 0);
    std::vector<float> hull_height(n + lower_hull.size() + upper_hull.size(),
                                   0.);
    std::vector<int> hull_vertices(n + lower_hull.size() + upper_hull.size(),
                                   0);

    for (long int i = 0; i < n; i++)
    {
        ff[i] = img[i * stride];
    }

    long int loop_start = 1;

    std::vector<float> ranges(n + lower_hull.size() + upper_hull.size() + 1, 0);
    ranges[0] = -INFINITY;
    ranges[1] = +INFINITY;

    int k = 0;
    if (lower_hull.size() > 0)
    {
        loop_start = 0;
        for (auto it = lower_hull.rbegin(); it != lower_hull.rend(); ++it)
        {
            const Hull& h = *it;
            hull_vertices[k] = h.vertex;
            hull_height[k] = h.height;
            ranges[k] = h.range;
            ranges[k + 1] = +INFINITY;
            k++;
        }
        k--;
    }
    else
    {
        hull_vertices[0] = 0;
        hull_height[0] = ff[0];
    }

    for (long int i = loop_start; i < n; i++)
    {
        update_hull(
            k, i, ff[i], hull_vertices, hull_height, ranges, anisotropy_factor);
    }

    // Upper corrector - add n to upper vertices as they were passed in with
    // relative locations
    if (upper_hull.size() > 0)
    {
        for (const Hull& h : upper_hull)
        {
            update_hull(k,
                        h.vertex + n,
                        h.height,
                        hull_vertices,
                        hull_height,
                        ranges,
                        anisotropy_factor);
        }
    }

    k = 0;

    for (long int i = 0; i < n; i++)
    {
        while (ranges[k + 1] < i)
        {
            k++;
        }
        img[i * stride] =
            anisotropy_factor * sq(i - hull_vertices[k]) + hull_height[k];
    }

    return;
}

/**
 * @brief Computes and returns a vector of boundary hulls based on the input
 * image data.
 *
 *
 * @param img Pointer to a 1D array of image data, representing pixel intensity
 * values.
 * @param resolution The image resolution in dimension
 * @param n The number of elements (pixels) in the image data.
 * @param stride The stride between consecutive elements in the image data
 * array.
 * @param num_hull The number of hulls to return. If this exceeds the calculated
 * hulls, it is capped.
 * @param left Boolean indicating the direction:
 *             - `true`: Return leftmost hulls.
 *             - `false`: Return rightmost hulls.
 *
 * @return A vector of `Hull` structures, each containing:
 *         - `vertices`: Vertex index of the hull.
 *         - `height`: Height of the hull at the vertex.
 *         - `range`: Range of values covered by the hull.
 */
std::vector<Hull>
return_boundary_hull(float* img,
                     const int n,
                     const float resolution,
                     const long int stride,
                     int num_hull,
                     const int index_corrector,
                     bool forward)
{
    to_finite(img, n);
    std::vector<Hull> hull;

    if (n == 1 && img[0] < std::numeric_limits<float>::max() - 1)
    {
        hull.push_back({ index_corrector, img[0], INFINITY });
        return hull;
    }

    const float anisotropy_factor = resolution * resolution;
    std::vector<float> ff(n, 0);
    std::vector<float> hull_height(n, 0);
    std::vector<int> hull_vertices(n, 0);

    for (long int i = 0; i < n; i++)
    {
        ff[i] = img[i * stride];
    }

    hull_height[0] = ff[0];
    std::vector<float> ranges(n + 1, 0);
    ranges[0] = -INFINITY;
    ranges[1] = +INFINITY;

    int k = 1;
    for (long int i = 1; i < n; i++)
    {
        update_hull(
            k, i, ff[i], hull_vertices, hull_height, ranges, anisotropy_factor);
    }

    num_hull = std::min(num_hull, k);
    hull.reserve(num_hull); // Preallocate memory

    // Select hulls
    int kk = forward ? 0 : k;
    int step = forward ? 1 : -1;
    while (static_cast<int>(hull.size()) <= num_hull && kk >= 0 && kk <= k)
    {
        if (hull_height[kk] < (std::numeric_limits<float>::max() - 1))
        {
            hull.push_back({ hull_vertices[kk] + index_corrector,
                             hull_height[kk],
                             ranges[kk] + index_corrector });
            if (hull_height[kk] == 0.9f)
                break; // Stop early for minimal parabolas
        }
        kk += step;
    }

    return hull;
}

#endif