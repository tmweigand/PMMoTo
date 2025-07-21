#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>

struct coordinate
{
    int x;
    int y;
    int z;
    int ID;
    int faceCount;
    float edt;
};

inline int
flat_index(int x, int y, int z, const size_t* shape)
{
    return z * shape[1] * shape[0] + y * shape[0] + x;
}

constexpr std::array<int, 256>
make_Euler_LUT()
{
    constexpr std::array<int, 128> arr = {
        1,  -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
        -3, -1, 3,  1, 1,  -1, 3,  1, -1, 1, 1, -1, 3, 1, 1, -1,
        -3, 3,  -1, 1, 1,  3,  -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
        1,  3,  3,  1, 5,  3,  3,  1, -1, 1, 1, -1, 3, 1, 1, -1,
        -7, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
        -3, -1, 3,  1, 1,  -1, 3,  1, -1, 1, 1, -1, 3, 1, 1, -1,
        -3, 3,  -1, 1, 1,  3,  -1, 1, -1, 1, 1, -1, 3, 1, 1, -1,
        1,  3,  3,  1, 5,  3,  3,  1, -1, 1, 1, -1, 3, 1, 1, -1
    };

    std::array<int, 256> LUT{};
    for (size_t i = 0; i < 128; ++i)
    {
        LUT[1 + 2 * i] = arr[i]; // odd indices: 1, 3, 5, ..., 255
    }
    return LUT;
}

constexpr auto Euler_LUT = make_Euler_LUT();

constexpr std::array<std::array<int, 7>, 8> neighbor_indices = { {
    // Octants:
    { { 2, 1, 11, 10, 5, 4, 14 } },     // NEB
    { { 0, 9, 3, 12, 1, 10, 4 } },      // NWB
    { { 8, 7, 17, 16, 5, 4, 14 } },     // SEB
    { { 6, 15, 7, 16, 3, 12, 4 } },     // SWB
    { { 20, 23, 19, 22, 11, 14, 10 } }, // NEU
    { { 18, 21, 9, 12, 19, 22, 10 } },  // NWU
    { { 26, 23, 17, 14, 25, 22, 16 } }, // SEU
    { { 24, 25, 15, 16, 21, 22, 12 } }, // SWU
} };

/**
 * @brief Extract selected voxels from the 3D neighborhood of a voxel.
 * @param img           Flat pointer to 3D image data stored as [z][y][x] in
 * row-major order.
 * @param x             X coordinate of the center voxel.
 * @param y             Y coordinate of the center voxel.
 * @param z             Z coordinate of the center voxel.
 * @param shape         Array of 3 elements: shape[0] = X, shape[1] = Y,
 * shape[2] = Z.
 * @param index       Array of corresponding to cube feature such as (-1,1,1).
 * @return std::vector<uint8_t> of the extracted neighbor values.
 */
std::vector<uint8_t>
get_neighborhood(const uint8_t* img,
                 int x,
                 int y,
                 int z,
                 const size_t* shape,
                 const std::vector<int>& index = { 0, 0, 0 })
{
    std::vector<uint8_t> neighbors(27, 0);
    int idx;

    int dx_min = (index[0] == -1) ? 0 : -1;
    int dx_max = (index[0] == 1) ? 0 : 1;

    int dy_min = (index[1] == -1) ? 0 : -1;
    int dy_max = (index[1] == 1) ? 0 : 1;

    int dz_min = (index[2] == -1) ? 0 : -1;
    int dz_max = (index[2] == 1) ? 0 : 1;

    std::cout << "Computing neighborhood bounds with index: [" << index[0]
              << ", " << index[1] << ", " << index[2] << "]" << std::endl;

    std::cout << "dx_min = " << dx_min << ", dx_max = " << dx_max << std::endl;
    std::cout << "dy_min = " << dy_min << ", dy_max = " << dy_max << std::endl;
    std::cout << "dz_min = " << dz_min << ", dz_max = " << dz_max << std::endl;

    // Following skeletonize implementation for now
    for (int dz = dz_min; dz <= dz_max; ++dz)
    {
        for (int dx = dx_min; dx <= dx_max; ++dx)
        {
            for (int dy = dy_min; dy <= dy_max; ++dy)
            {
                int xi = x + dx;
                int yi = y + dy;
                int zi = z + dz;
                idx = (dz + 1) * 9 + (dx + 1) * 3 + (dy + 1);
                neighbors[idx] = img[flat_index(xi, yi, zi, shape)];
            }
        }
    }
    return neighbors;
}

bool
is_endpoint(const std::vector<uint8_t>& neighbors)
{
    // An endpoint has exactly one neighbor in the 26-neighborhood,
    // meaning sum == 2 (1 for center pixel + 1 neighbor)
    int sum = std::accumulate(neighbors.begin(), neighbors.end(), 0);
    return sum == 2;
}

// neighbors : must be size 27 lut : must be size 256
inline bool
is_Euler_invariant(
    const std::vector<uint8_t>& neighbors,
    const std::vector<int>& octants = { 0, 1, 2, 3, 4, 5, 6, 7 }) noexcept
{
    int euler_char = 0;

    for (int octant : octants)
    {
        int n = 1;
        for (int j = 0; j < 7; ++j)
        {
            int idx = neighbor_indices[octant][j];
            if (neighbors[idx] == 1)
            {
                n |= (1 << (7 - j));
            }
        }
        euler_char += Euler_LUT[n];
    }

    return euler_char == 0;
}

static const std::array<std::array<std::pair<int, std::vector<int> >, 7>, 8>
    octree = { { // Octant 1
                 { { { 0, {} },
                     { 1, { 2 } },
                     { 3, { 3 } },
                     { 4, { 2, 3, 4 } },
                     { 9, { 5 } },
                     { 10, { 2, 5, 6 } },
                     { 12, { 3, 5, 7 } } } },
                 // Octant 2
                 { { { 1, { 1 } },
                     { 4, { 1, 3, 4 } },
                     { 10, { 1, 5, 6 } },
                     { 2, {} },
                     { 5, { 4 } },
                     { 11, { 6 } },
                     { 13, { 4, 6, 8 } } } },
                 // Octant 3
                 { { { 3, { 1 } },
                     { 4, { 1, 2, 4 } },
                     { 12, { 1, 5, 7 } },
                     { 6, {} },
                     { 7, { 4 } },
                     { 14, { 7 } },
                     { 15, { 4, 7, 8 } } } },
                 // Octant 4
                 { { { 4, { 1, 2, 3 } },
                     { 5, { 2 } },
                     { 13, { 2, 6, 8 } },
                     { 7, { 3 } },
                     { 15, { 3, 7, 8 } },
                     { 8, {} },
                     { 16, { 8 } } } },
                 // Octant 5
                 { { { 9, { 1 } },
                     { 10, { 1, 2, 6 } },
                     { 12, { 1, 3, 7 } },
                     { 17, {} },
                     { 18, { 6 } },
                     { 20, { 7 } },
                     { 21, { 6, 7, 8 } } } },
                 // Octant 6
                 { { { 10, { 1, 2, 5 } },
                     { 11, { 2 } },
                     { 13, { 2, 4, 8 } },
                     { 18, { 5 } },
                     { 21, { 5, 7, 8 } },
                     { 19, {} },
                     { 22, { 8 } } } },
                 // Octant 7
                 { { { 12, { 1, 3, 5 } },
                     { 14, { 3 } },
                     { 15, { 3, 4, 8 } },
                     { 20, { 5 } },
                     { 21, { 5, 6, 8 } },
                     { 23, {} },
                     { 24, { 8 } } } },
                 // Octant 8
                 { { { 13, { 2, 4, 6 } },
                     { 15, { 3, 4, 7 } },
                     { 16, { 4 } },
                     { 21, { 5, 6, 7 } },
                     { 22, { 6 } },
                     { 24, { 7 } },
                     { 25, {} } } } } };

/**
 * @brief Perform recursive octree-based labeling of connected voxels in a local
 * 3D neighborhood.
 *
 * This function labels connected components in a voxel neighborhood by
 * octant-based recursion. Based on the N(v)-labeling approach from Lee et al.
 * (1994), using pre-defined connectivity rules.
 *
 * @param octant The current octant index (1 to 8).
 * @param label The current label to assign.
 * @param cube A flat array of 26 uint8_t values representing the neighborhood,
 * excluding the center.
 */
void
octree_labeling(int octant, int label, std::array<uint8_t, 26>& cube)
{
    for (const auto& [voxel_idx, recurse_octants] : octree[octant - 1])
    {
        if (cube[voxel_idx] == 1)
        {
            cube[voxel_idx] = label;
            for (int new_octant : recurse_octants)
            {
                octree_labeling(new_octant, label, cube);
            }
        }
    }
}

/**
 * @brief Check whether a point is a simple point in a 3x3x3 voxel neighborhood.
 *
 * A voxel is considered simple if its deletion does not change the connectivity
 * of the foreground in its 26-neighbor 3D neighborhood. This implements the
 * N(v)_labeling method described in Lee et al. (1994).
 *
 * @param neighbors A pointer to a 27-element array representing a 3x3x3
 * neighborhood of voxel values (typically 0 or 1), ordered in raster scan
 * (z-fastest). The center voxel is at index 13 and is ignored during
 * evaluation.
 *
 * @return true if the voxel is a simple point (i.e., its deletion preserves
 * topology), false otherwise.
 *
 * @note This function assumes that `octree_labeling(int octant, int label,
 * pixel_type cube[26])` is defined elsewhere and performs recursive labeling
 * within the given octant.
 */
bool
is_simple_point(const std::vector<uint8_t>& neighbors)
{
    std::array<uint8_t, 26> cube;
    // Copy neighbors[0..12] into cube[0..12]
    std::memcpy(cube.data(), neighbors.data(), 13 * sizeof(uint8_t));

    // Copy neighbors[14..26] into cube[13..25], skipping center (index 13)
    std::memcpy(cube.data() + 13, neighbors.data() + 14, 13 * sizeof(uint8_t));

    int label = 2;

    // Lookup table mapping voxel index (0–25, excluding center) to octant
    // number (1–8)
    static constexpr std::array<uint8_t, 26> voxel_to_octant = {
        { 1, 1, 2, 1, 1, 2, 3, 3, 4, 1, 1, 2, 1,
          2, 3, 3, 4, 5, 5, 6, 5, 5, 6, 7, 7, 8 }
    };

    for (int i = 0; i < 26; ++i)
    {
        if (cube[i] == 1)
        {
            int octant = voxel_to_octant[i];
            octree_labeling(octant, label, cube);
            label++;
            if (label - 2 >= 2)
            {
                return false;
            }
        }
    }

    return true;
}

std::vector<coordinate>
find_simple_points(const uint8_t* img,
                   const float* edt,
                   const size_t* shape,
                   const std::vector<std::pair<int, int> >& loop)
{
    coordinate point;
    std::vector<coordinate> simple_border_points;

    for (int x = loop[0].first; x < loop[0].second; ++x)
    {
        for (int y = loop[1].first; y < loop[1].second; ++y)
        {
            for (int z = loop[2].first; z < loop[2].second; ++z)
            {
                if (img[flat_index(x, y, z, shape)] != 1) continue;

                auto neighbors = get_neighborhood(img, x, y, z, shape);

                std::cout << is_endpoint(neighbors) << " "
                          << is_Euler_invariant(neighbors) << " "
                          << is_simple_point(neighbors) << std::endl;

                if (is_endpoint(neighbors) || !is_Euler_invariant(neighbors) ||
                    !is_simple_point(neighbors))
                {
                    continue;
                }

                point.x = x;
                point.y = y;
                point.z = z;
                point.ID = 0;
                point.edt = edt[flat_index(x, y, z, shape)];

                simple_border_points.push_back(point);
            }
        }
    }
    return simple_border_points;
}

bool
is_foreground(const uint8_t* img, int x, int y, int z, const size_t* shape)
{
    if (x < 0 || y < 0 || z < 0 || x >= shape[0] || y >= shape[1] ||
        z >= shape[2])
        return false; // out-of-bounds treated as background
    return img[flat_index(x, y, z, shape)] == 1;
}

void
find_simple_point_candidates(uint8_t* img,
                             int curr_border,
                             std::vector<coordinate>& simple_border_points,
                             const std::vector<std::pair<int, int> >& loop,
                             const size_t* shape,
                             const std::vector<int>& index = { 0, 0, 0 })
{
    for (int x = loop[0].first; x < loop[0].second; ++x)
    {
        for (int y = loop[1].first; y < loop[1].second; ++y)
        {
            for (int z = loop[2].first; z < loop[2].second; ++z)
            {
                if (img[flat_index(x, y, z, shape)] != 1) continue;

                bool is_border_pt =
                    (curr_border == 1 &&
                     !is_foreground(img, x, y, z - 1, shape)) || // N
                    (curr_border == 2 &&
                     !is_foreground(img, x, y, z + 1, shape)) || // S
                    (curr_border == 3 &&
                     !is_foreground(img, x, y + 1, z, shape)) || // E
                    (curr_border == 4 &&
                     !is_foreground(img, x, y - 1, z, shape)) || // W
                    (curr_border == 5 &&
                     !is_foreground(img, x + 1, y, z, shape)) || // U
                    (curr_border == 6 &&
                     !is_foreground(img, x - 1, y, z, shape)); // B

                if (!is_border_pt) continue;

                auto neighborhood =
                    get_neighborhood(img, x, y, z, shape, index);

                if (is_endpoint(neighborhood)) continue;
                if (!is_Euler_invariant(neighborhood)) continue;
                if (!is_simple_point(neighborhood)) continue;

                coordinate point;
                point.x = x;
                point.y = y;
                point.z = z;
                point.ID = 0;

                simple_border_points.push_back(point);
            }
        }
    }
}

void
compute_thin_image(uint8_t* img, const size_t* shape)
{
    std::array<int, 6> borders = { 4, 3, 2, 1, 5, 6 };

    int unchanged_borders = 0;
    int num_borders = 6;
    std::vector<coordinate> simple_border_points;

    while (unchanged_borders < num_borders)
    {
        unchanged_borders = 0;

        for (int j = 0; j < num_borders; ++j)
        {
            int curr_border = borders[j];
            simple_border_points.clear();

            // find_simple_point_candidates(
            //     img, curr_border, simple_border_points, shape);

            // // Sort by faceCount descending
            // std::sort(simple_border_points.begin(),
            //           simple_border_points.end(),
            //           [](const Coordinate& a, const Coordinate& b)
            //           { return a.faceCount > b.faceCount; });

            bool no_change = true;
            for (const auto& point : simple_border_points)
            {
                int x = point.x;
                int y = point.y;
                int z = point.z;
                int ID = point.ID;

                auto neighbors = get_neighborhood(img, x, y, z, shape);

                if (is_simple_point(neighbors))
                {
                    img[flat_index(x, y, z, shape)] = 0;
                    no_change = false;
                }
            }

            if (no_change) ++unchanged_borders;
        }
    }
}
