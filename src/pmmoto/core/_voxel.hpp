
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>

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

int64_t _get_nearest_boundary_index(uint8_t *img, uint8_t label, const int n,
                                    const long int stride,
                                    bool forward = true) {
  /*
   Finds the nearest occurrence of `label` in the provided 1D slice of the
   image.

   Parameters:
       img (uint8_t*): Pointer to the 1D slice of the image.
       label (uint8_t): The label to find.
       n (int): Number of elements in the 1D slice.
       stride ( const long int): Memory stride between elements in the slice.
       forward (bool): Search direction (true for forward, false for backward).

   Returns:
       int64_t: The index (0-based) of the nearest occurrence of `label`, or -1
   if not found.
  */

  // Determine iteration parameters based on direction
  int64_t start = forward ? 0 : (n - 1) * stride;
  int64_t end = forward ? n * stride : -stride;
  int64_t step = forward ? stride : -stride;

  // Iterate through the slice to find the label
  for (int64_t i = start, count = 0; i != end; i += step, count++) {
    if (img[i] == label) {
      return forward ? count : (n - 1) - count;
    }
  }

  return -1; // Return -1 if the label is not found
}