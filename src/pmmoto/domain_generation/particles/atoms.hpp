#ifndef ATOMS_H
#define ATOMS_H

#include "particle_list.hpp"

/**
 * @class Atom
 * @brief Represents a sphere with position and radius.
 */
class Atom : public Particle
{
public:
    Atom(double x, double y, double z, double radius)
        : Particle({ x, y, z }, radius)
    {
    }
    Atom(std::vector<double> coordinates, double radius)
        : Particle(coordinates, radius)
    {
    }
    Atom(std::vector<double> atom_data) : Particle(atom_data, atom_data[3])
    {
    }
};

class AtomList : public ParticleList<Atom>
{
public:
    AtomList(std::vector<Atom> atom_data) : ParticleList<Atom>(atom_data)
    {
    }
};

#endif
