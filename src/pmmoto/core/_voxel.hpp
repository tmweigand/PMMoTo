
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

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