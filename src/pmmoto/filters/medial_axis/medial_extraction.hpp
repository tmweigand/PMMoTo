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
    std::vector<int> index = { 0, 0, 0 };
    float edt;
    std::vector<int> vertices = {};
};

inline int
flat_index(int x, int y, int z, const size_t* shape)
{
    return x * shape[1] * shape[2] + y * shape[2] + z;
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
    { { 0, 1, 3, 4, 9, 10, 12 } },
    { { 2, 1, 5, 4, 11, 10, 14 } },
    { { 6, 7, 3, 4, 15, 16, 12 } },
    { { 8, 7, 5, 4, 17, 16, 14 } },
    { { 18, 19, 21, 22, 9, 10, 12 } },
    { { 20, 19, 23, 22, 11, 10, 14 } },
    { { 24, 25, 21, 22, 15, 16, 12 } },
    { { 26, 25, 23, 22, 17, 16, 14 } },
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

    for (int dx = dx_min; dx <= dx_max; ++dx)
    {
        for (int dy = dy_min; dy <= dy_max; ++dy)
        {
            for (int dz = dz_min; dz <= dz_max; ++dz)
            {
                int xi = x + dx;
                int yi = y + dy;
                int zi = z + dz;
                idx = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1);
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
    octree = { {
        { // Octant 0
          std::make_pair(0, std::vector<int>{}),
          std::make_pair(1, std::vector<int>{ 1 }),
          std::make_pair(3, std::vector<int>{ 2 }),
          std::make_pair(4, std::vector<int>{ 1, 2, 3 }),
          std::make_pair(9, std::vector<int>{ 4 }),
          std::make_pair(10, std::vector<int>{ 1, 4, 5 }),
          std::make_pair(12, std::vector<int>{ 2, 4, 6 }) },
        { // Octant 1
          std::make_pair(2, std::vector<int>{}),
          std::make_pair(5, std::vector<int>{ 3 }),
          std::make_pair(11, std::vector<int>{ 5 }),
          std::make_pair(13, std::vector<int>{ 3, 5, 7 }),
          std::make_pair(1, std::vector<int>{ 0 }),
          std::make_pair(4, std::vector<int>{ 0, 2, 3 }),
          std::make_pair(10, std::vector<int>{ 0, 4, 5 }) },
        { // Octant 2
          std::make_pair(6, std::vector<int>{}),
          std::make_pair(7, std::vector<int>{ 3 }),
          std::make_pair(14, std::vector<int>{ 6 }),
          std::make_pair(15, std::vector<int>{ 3, 6, 7 }),
          std::make_pair(3, std::vector<int>{ 0 }),
          std::make_pair(4, std::vector<int>{ 0, 1, 3 }),
          std::make_pair(12, std::vector<int>{ 0, 4, 6 }) },
        { // Octant 3
          std::make_pair(8, std::vector<int>{}),
          std::make_pair(16, std::vector<int>{ 7 }),
          std::make_pair(7, std::vector<int>{ 2 }),
          std::make_pair(15, std::vector<int>{ 2, 6, 7 }),
          std::make_pair(5, std::vector<int>{ 1 }),
          std::make_pair(13, std::vector<int>{ 1, 5, 7 }),
          std::make_pair(4, std::vector<int>{ 0, 1, 2 }) },
        { // Octant 4
          std::make_pair(17, std::vector<int>{}),
          std::make_pair(18, std::vector<int>{ 5 }),
          std::make_pair(20, std::vector<int>{ 6 }),
          std::make_pair(21, std::vector<int>{ 5, 6, 7 }),
          std::make_pair(9, std::vector<int>{ 0 }),
          std::make_pair(10, std::vector<int>{ 0, 1, 5 }),
          std::make_pair(12, std::vector<int>{ 0, 2, 6 }) },
        { // Octant 5
          std::make_pair(19, std::vector<int>{}),
          std::make_pair(22, std::vector<int>{ 7 }),
          std::make_pair(18, std::vector<int>{ 4 }),
          std::make_pair(21, std::vector<int>{ 4, 6, 7 }),
          std::make_pair(11, std::vector<int>{ 1 }),
          std::make_pair(13, std::vector<int>{ 1, 3, 7 }),
          std::make_pair(10, std::vector<int>{ 0, 1, 4 }) },
        { // Octant 6
          std::make_pair(23, std::vector<int>{}),
          std::make_pair(24, std::vector<int>{ 7 }),
          std::make_pair(20, std::vector<int>{ 4 }),
          std::make_pair(21, std::vector<int>{ 4, 5, 7 }),
          std::make_pair(14, std::vector<int>{ 2 }),
          std::make_pair(15, std::vector<int>{ 2, 3, 7 }),
          std::make_pair(12, std::vector<int>{ 0, 2, 4 }) },
        { // Octant 7
          std::make_pair(25, std::vector<int>{}),
          std::make_pair(24, std::vector<int>{ 6 }),
          std::make_pair(22, std::vector<int>{ 5 }),
          std::make_pair(21, std::vector<int>{ 4, 5, 6 }),
          std::make_pair(16, std::vector<int>{ 3 }),
          std::make_pair(15, std::vector<int>{ 2, 3, 6 }),
          std::make_pair(13, std::vector<int>{ 1, 3, 5 }) },
    } };

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
    for (const auto& [voxel_idx, recurse_octants] : octree[octant])
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
 *
 * @note: If only the center voxel is present, most implementations return true
 * which should not be the case, as it deletes an object, thus changes the
 * connectivity. For this, a check for this conditions is performed and false is
 * returned.
 *
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
    bool is_single =
        std::accumulate(neighbors.begin(), neighbors.end(), 0) == 1;
    if (is_single) return false;

    std::array<uint8_t, 26> cube;
    // Copy neighbors[0..12] into cube[0..12]
    std::memcpy(cube.data(), neighbors.data(), 13 * sizeof(uint8_t));

    // Copy neighbors[14..26] into cube[13..25], skipping center (index 13)
    std::memcpy(cube.data() + 13, neighbors.data() + 14, 13 * sizeof(uint8_t));

    int label = 2;

    static constexpr std::array<uint8_t, 26> voxel_octant_owner = {
        { 0, 0, 1, 0, 0, 1, 2, 2, 3, 0, 0, 1, 0,
          1, 2, 2, 3, 4, 4, 5, 4, 4, 5, 6, 6, 7 }
    };

    for (int i = 0; i < 26; ++i)
    {
        if (cube[i] == 1)
        {
            int octant = voxel_octant_owner[i];
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

bool(const uint8_t* img, int x, int y, int z, const size_t* shape)
{
    if (x < 0 || y < 0 || z < 0 || x > shape[0] - 1 || y > shape[1] - 1 ||
        z > shape[2] - 1)
        return true; // out-of-bounds treated as background

    return img[flat_index(x, y, z, shape)] == 0;
}

bool
is_last_boundary_point(const std::vector<uint8_t>& neighbors,
                       const std::vector<int>& vertices)
{
    int sum = 0;
    for (int v : vertices)
    {
        sum += neighbors[v];
    }
    return sum == 1;
}

void
find_boundary_simple_point_candidates(
    uint8_t* img,
    const std::vector<int>& erode_index,
    std::vector<coordinate>& simple_border_points,
    const std::vector<std::pair<int, int> >& loop,
    const size_t* shape,
    const std::vector<int>& index,
    const std::vector<int>& octants,
    const std::vector<int>& vertices)
{
    for (int x = loop[0].first; x < loop[0].second; ++x)
    {
        for (int y = loop[1].first; y < loop[1].second; ++y)
        {
            for (int z = loop[2].first; z < loop[2].second; ++z)
            {
                if (img[flat_index(x, y, z, shape)] != 1) continue;

                bool is_border_pt = is_foreground(img,
                                                  x + erode_index[0],
                                                  y + erode_index[1],
                                                  z + erode_index[2],
                                                  shape);

                if (!is_border_pt) continue;

                // This performs bounds checks with index
                auto neighborhood =
                    get_neighborhood(img, x, y, z, shape, index);

                if (is_last_boundary_point(neighborhood, vertices)) continue;

                if (is_endpoint(neighborhood)) continue;
                if (!is_Euler_invariant(neighborhood, octants)) continue;
                if (!is_simple_point(neighborhood)) continue;

                // std::cout << "Simple Point" << std::endl;
                coordinate point;
                point.x = x;
                point.y = y;
                point.z = z;
                point.index = index;
                point.vertices = vertices;

                simple_border_points.push_back(point);
            }
        }
    }
}

bool
remove_points(uint8_t* img,
              std::vector<coordinate>& simple_points,
              const size_t* shape)
{
    bool no_change = true;
    int count_change = 0;

    for (const auto& point : simple_points)
    {
        int x = point.x;
        int y = point.y;
        int z = point.z;
        const auto& index = point.index;
        const auto& vertices = point.vertices;

        auto neighbors = get_neighborhood(img, x, y, z, shape, index);

        if (!(index[0] == 0 && index[1] == 0 && index[2] == 0))
        {
            if (is_last_boundary_point(neighbors, vertices))
            {
                continue;
            }
        }

        if (is_simple_point(neighbors))
        {
            img[flat_index(x, y, z, shape)] = 0;
            no_change = false;
            count_change++;
        }
    }

    return no_change;
}

void
find_simple_point_candidates(uint8_t* img,
                             const std::vector<int>& erode_index,
                             std::vector<coordinate>& simple_border_points,
                             const std::vector<std::pair<int, int> >& loop,
                             const size_t* shape)
{
    for (int x = loop[0].first; x < loop[0].second; ++x)
    {
        for (int y = loop[1].first; y < loop[1].second; ++y)
        {
            for (int z = loop[2].first; z < loop[2].second; ++z)
            {
                if (img[flat_index(x, y, z, shape)] != 1) continue;

                bool is_border_pt = is_foreground(img,
                                                  x + erode_index[0],
                                                  y + erode_index[1],
                                                  z + erode_index[2],
                                                  shape);

                if (!is_border_pt) continue;

                // This performs bounds checks with index
                auto neighborhood = get_neighborhood(img, x, y, z, shape);

                if (is_endpoint(neighborhood)) continue;
                if (!is_Euler_invariant(neighborhood)) continue;
                if (!is_simple_point(neighborhood)) continue;

                coordinate point;
                point.x = x;
                point.y = y;
                point.z = z;

                simple_border_points.push_back(point);
            }
        }
    }
}
