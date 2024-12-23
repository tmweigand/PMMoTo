#ifndef DISTANCE_H
#define DISTANCE_H

#include "threadpool.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#define sq(x) (static_cast<float>(x) * static_cast<float>(x))

inline void to_finite(float *f, const size_t voxels) {
  for (size_t i = 0; i < voxels; i++) {
    if (f[i] == INFINITY) {
      f[i] = std::numeric_limits<float>::max() - 1;
    }
  }
}

inline void to_infinite(float *f, const size_t voxels) {
  for (size_t i = 0; i < voxels; i++) {
    if (f[i] >= std::numeric_limits<float>::max() - 1) {
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
inline float intersect_parabolas(const float f_r, const float r,
                                 const float f_q, const float q,
                                 const float anistropy_factor) {
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
 * @param w2             Constant parameter for the `intersect_parabolas`
 *                       function.
 *
 * This function computes the intersection of parabolas, updates the hull
 * structure by maintaining the convexity property, and adjusts the range
 * boundaries accordingly.
 */
void update_hull(int &k, int i, float ff, std::vector<int> &hull_vertices,
                 std::vector<float> &hull_height, std::vector<float> &ranges,
                 float w2) {

  float s = intersect_parabolas(ff, i, hull_height[k], hull_vertices[k], w2);

  while (k > 0 && s <= ranges[k]) {
    k--;
    s = intersect_parabolas(ff, i, hull_height[k], hull_vertices[k], w2);
  }

  k++;
  hull_vertices[k] = i;
  hull_height[k] = ff;
  ranges[k] = s;
  ranges[k + 1] = INFINITY;
}

void _determine_boundary_parabolic_envelope(
    float *img, const int n, const long int stride, const int lower_vertex,
    const float lower_f, const int upper_vertex, const float upper_f) {

  if (n == 0) {
    return;
  }

  const float w2 = 1 * 1;

  std::vector<float> ff(n, 0);
  std::vector<float> hull_height(n + 2, 0.);
  std::vector<int> hull_vertices(n + 2, 0);

  for (long int i = 0; i < n; i++) {
    ff[i] = img[i * stride];
  }

  long int loop_start = 1;

  std::vector<float> ranges(n + 1, 0);
  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  if (lower_vertex != std::numeric_limits<int>::max()) {
    loop_start = 0;
    hull_vertices[0] = lower_vertex - n;
    hull_height[0] = lower_f;
  } else {
    hull_vertices[0] = 0;
    hull_height[0] = ff[0];
  }

  int k = 0;
  float s;
  for (long int i = loop_start; i < n; i++) {
    update_hull(k, i, ff[i], hull_vertices, hull_height, ranges, 1);
  }

  if (upper_vertex != std::numeric_limits<int>::max()) {
    update_hull(k, n + upper_vertex, upper_f, hull_vertices, hull_height,
                ranges, 1);
  }

  k = 0;
  for (long int i = 0; i < n; i++) {
    while (ranges[k + 1] < i) {
      k++;
    }
    img[i * stride] = w2 * sq(i - hull_vertices[k]) + hull_height[k];
  }

  return;
}

/**
 * @brief Computes the squared Euclidean distance transform (EDT) for a 1D array
 * with multiple segments.
 *
 * This function calculates the squared Euclidean distance transform for a
 * segmented 1D array. It processes segments independently, applying specified
 * anisotropy and correction factors.
 *
 * @tparam T Type of the segment identifiers.
 * @param[in] segids Pointer to the array of segment identifiers. A value of 0
 * indicates no segment.
 * @param[out] d Pointer to the distance array, which will be updated with
 * squared distances.
 * @param[in] n Number of elements in the array.
 * @param[in] stride Stride value used for indexing into the arrays (useful for
 * multi-dimensional arrays).
 * @param[in] anistropy The anisotropy factor applied to distances between
 * elements.
 * @param[in] lower_corrector Distance correction factor for transitions into a
 * segment. Defaults to `INFINITY`.
 * @param[in] upper_corrector Maximum allowed distance value. Defaults to
 * `INFINITY`.
 */
template <typename T>
void squared_edt_1d_multi_seg_new(T *segids, float *d, const int n,
                                  const long int stride, const float anistropy,
                                  const float lower_corrector = INFINITY,
                                  const float upper_corrector = INFINITY) {

  long int i;
  T working_segid = segids[0];
  d[0] = working_segid == 0 ? 0 : lower_corrector;

  for (i = stride; i < n * stride; i += stride) {

    if (segids[i] == 0) {
      d[i] = 0.0;
    } else if (segids[i] == working_segid) {
      d[i] = d[i - stride] + anistropy;
    } else {
      d[i] = anistropy;
      d[i - stride] = static_cast<float>(segids[i - stride] != 0) * anistropy;
      working_segid = segids[i];
    }
  }

  if (d[(n - 1) * stride] > upper_corrector) {
    d[(n - 1) * stride] = upper_corrector;
  }

  for (i = (n - 2) * stride; i >= 0; i -= stride) {
    d[i] = std::fminf(d[i], d[i + stride] + anistropy);
  }

  for (i = 0; i < n * stride; i += stride) {
    d[i] *= d[i];
  }
}

void return_boundary_hull(float *img, const int n, const long int stride,
                          int &hull_vertex, float &hull_f, bool left) {

  to_finite(img, n);
  if (n == 0) {
    return;
  }

  const float w2 = 1 * 1;
  std::vector<float> ff(n, 0);
  std::vector<float> hull_height(n, 0);
  std::vector<int> hull_vertices(n, 0);

  for (long int i = 0; i < n; i++) {
    ff[i] = img[i * stride];
  }
  hull_height[0] = ff[0];
  std::vector<float> ranges(n + 1, 0);
  ranges[0] = -INFINITY;
  ranges[1] = +INFINITY;

  float s;
  int k = 0;
  for (long int i = 1; i < n; i++) {
    update_hull(k, i, ff[i], hull_vertices, hull_height, ranges, 1);
  }

  k = 0;
  // Find the first valid range
  if (left) {
    while (ranges[k + 1] < 0) {
      ++k;
    }
    hull_vertex = hull_vertices[k];
    hull_f = hull_height[k];
  } else {
    // Find the last valid range
    while (ranges[k + 1] < n) {
      ++k;
    }
    hull_vertex = hull_vertices[k];
    hull_f = hull_height[k];
  }
  return;
}

#endif