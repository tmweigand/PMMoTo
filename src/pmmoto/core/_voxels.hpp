
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_set>

/**
 * @brief Iterates through a 1D slice of an image and writes the values to an
 * output array.
 *
 * This function traverses a 1D slice of an image based on the specified stride
 * and direction, and copies the values into the output array. The first element
 * in the output array is initialized to a specific value depending on the
 * traversal direction.
 *
 * @param img Pointer to the 1D slice of the image.
 * @param out Pointer to the output array where the values will be written.
 * @param n The number of elements in the 1D slice.
 * @param stride Memory stride between elements in the slice.
 * @param forward Direction of traversal. If true, traverses forward; otherwise,
 * traverses backward.
 */
void loop_through_slice(uint8_t *img, uint8_t *out, const int n,
                        const long int stride, bool forward = true) {

  // Calculate start, end, and step, and initial value based on the direction
  int start = forward ? stride : (n - 2) * stride;
  int end = forward ? n * stride : -stride;
  int step = forward ? stride : -stride;
  uint8_t initial_value = forward ? img[0] : img[start - step];

  out[0] = initial_value;
  long int count = 1; // Start from the second element in 'out'
  for (int i = start; i != end; i += step) {
    out[count++] = img[i];
  }
}

/**
 * @brief Finds the nearest occurrence of a label in a 1D slice of an image.
 *
 * This function searches for the nearest occurrence of the specified `label`
 * in a 1D slice of the image. The search can be performed in either a forward
 * or backward direction.
 *
 * @param img Pointer to the 1D slice of the image.
 * @param label The label to find in the slice.
 * @param n The number of elements in the 1D slice.
 * @param stride Memory stride between elements in the slice.
 * @param index_corrector Same as img[start]. Ensures return wrt
 * img.shape[dimension]
 * @param forward Search direction. If true, searches forward; otherwise,
 * searches backward.
 * @return The 0-based index of the nearest occurrence of `label`, or -1 if not
 * found.
 */
int64_t _get_nearest_boundary_index(uint8_t *img, uint8_t label, const int n,
                                    const long int stride,
                                    const int index_corrector,
                                    bool forward = true) {

  // Determine iteration parameters based on direction
  int64_t start = forward ? 0 : (n - 1) * stride;
  int64_t end = forward ? n * stride : -stride;
  int64_t step = forward ? stride : -stride;

  // Iterate through the slice to find the label
  for (int64_t i = start, count = 0; i != end; i += step, count++) {
    if (img[i] == label) {
      return forward ? count + index_corrector
                     : (n - 1) - count + index_corrector;
    }
  }

  return -1; // Return -1 if the label is not found
}

/**
 * @brief Computes the unique pairs from a given array of pairs.
 *
 * This function takes an input array of pairs represented as a flat array
 * (alternating elements correspond to the first and second elements of each
 * pair) and returns a vector of unique pairs. The pairs are deduplicated using
 * a hash-based approach for efficiency.
 *
 * @param data Pointer to a flat array of uint64_t integers containing the
 * pairs. Each pair consists of two consecutive integers in the array.
 * @param nrows The number of pairs in the input array.
 *              This corresponds to half the size of the `data` array.
 * @return A `std::vector` of unique pairs represented as `std::pair<uint64_t,
 * uint64_t>`.
 *
 * @note
 * The input array should contain exactly `2 * nrows` elements.
 * The function uses an `std::unordered_set` for efficient deduplication.
 *
 * @example
 * uint64_t data[] = {1, 2, 2, 3, 1, 2, 3, 4};
 * size_t nrows = 4;
 * auto result = unique_pairs(data, nrows);
 * // Result: {{1, 2}, {2, 3}, {3, 4}}
 */
std::vector<std::pair<unsigned long, unsigned long>>
unique_pairs(unsigned long *data, size_t nrows) {
  std::unordered_set<unsigned long> unique_set;
  std::vector<std::pair<unsigned long, unsigned long>> result;

  // Encode each pair into a single 64-bit integer for uniqueness
  for (size_t i = 0; i < nrows; ++i) {
    unsigned long encoded = (data[2 * i] << 32) | data[2 * i + 1];
    unique_set.insert(encoded);
  }

  // Decode back into pairs
  for (unsigned long encoded : unique_set) {
    unsigned long first = encoded >> 32;         // Extract first element
    unsigned long second = encoded & 0xFFFFFFFF; // Extract second element
    result.emplace_back(first, second);
  }

  return result;
}