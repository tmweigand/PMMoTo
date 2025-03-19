#ifndef NEW_SPHERES_HPP
#define NEW_SPHERES_HPP

#include <iostream>

/**
 * @class SphereList
 * @brief Contains a list of the spheres and tools to operate on those spheres
 */
class NewSphereList
{
protected:
    // std::vector<Sphere> spheres;
    // ParticleList& particle_list;

public:
    // SphereList() = default;

    // // Define parameterized constructor
    // SphereList(std::vector<std::vector<double> > coordinates,
    //            std::vector<double> radii);

    NewSphereList();
    NewSphereList(double radii);
};

#endif // NEW_SPHERES_HPP