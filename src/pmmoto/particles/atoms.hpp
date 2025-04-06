#ifndef ATOMS_H
#define ATOMS_H

#include "spheres.hpp"

/**
 * @class Atom
 * @brief Represents a sphere with position and radius.
 */
class Atom : public Particle
{
public:
    double radius;

    Atom(double x, double y, double z, double radius)
        : Particle(x, y, z), radius(radius)
    {
    }
    Atom(std::vector<double> coordinates, double radius)
        : Particle(coordinates), radius(radius)
    {
    }
};

/**
 * @class AtomList
 * @brief Extends AtomList with additional operations for atoms.
 */
class AtomList : public SphereList
{
private:
public:
    using SphereList::SphereList; // Inherit constructors

    double radius;
    int label;
    double mass = 0;

    AtomList(std::vector<std::vector<double> > atom_coordinates,
             const double radius,
             const int label = 0,
             const double mass = 0)
        : SphereList(atom_coordinates, radius),
          radius(radius),
          label(label),
          mass(mass)
    {
    }

    std::vector<std::vector<double> > return_atoms(bool return_own = false,
                                                   bool return_label = false)
    {
        std::vector<std::vector<double> > info;
        info.reserve(spheres.size());
        for (const auto& sphere : spheres)
        {
            std::vector<double> _info = { sphere.coordinates[0],
                                          sphere.coordinates[1],
                                          sphere.coordinates[2],
                                          sphere.radius };

            if (return_own) _info.push_back(static_cast<double>(sphere.own));
            if (return_label) _info.push_back(static_cast<double>(label));

            info.emplace_back(std::move(_info));
        }

        return info;
    }

    void add_periodic_atoms(Box& domain, const Box& subdomain)
    {
        add_periodic_spheres(domain, subdomain);
    }

    /**
     * @brief Removes atoms that are not within nor intersect specified box
     */
    void trim_atoms_intersecting(const Box& subdomain)
    {
        trim_spheres_intersecting(subdomain);
    }

    /**
     * @brief Removes atoms that are not within specified box
     */
    void trim_atoms_within(const Box& subdomain)
    {
        trim_spheres_within(subdomain);
    }

    void set_own_atoms(const Box& subdomain)
    {
        own_spheres(subdomain);
    }

    size_t get_atom_count()
    {
        auto own_atoms = get_own_count();
        return own_atoms;
        ;
    }
};

/**
 * @brief Groups atom coordinates by their respective atom types.
 *
 * @param atom_coordinates A vector of 3D coordinates, where each entry
 * represents an atom's position.
 * @param atom_ids A vector containing atom type identifiers, corresponding to
 * each atom in `atom_coordinates`.
 * @return A map where each unique atom type is associated with a
 * vector of its corresponding coordinates.
 */
std::unordered_map<int, std::vector<std::vector<double> > >
group_atoms_by_type(const std::vector<std::vector<double> >& atom_coordinates,
                    const std::vector<int>& atom_ids)
{
    std::unordered_map<int, std::vector<std::vector<double> > > atoms_by_type;

    size_t num_atoms = atom_ids.size();

    // Reserve space for efficiency if atom types are known
    atoms_by_type.reserve(num_atoms);

    for (size_t i = 0; i < num_atoms; ++i)
    {
        atoms_by_type[atom_ids[i]].emplace_back(atom_coordinates[i]);
    }

    return atoms_by_type;
}

/**
 * @brief Determine the values based on atom type
 */
std::vector<double>
atom_id_to_values(const std::vector<int>& atom_ids,
                  const std::unordered_map<int, double>& value)
{
    // Create a vector to store radius values, same size as atom_ids
    std::vector<double> _value(atom_ids.size());

    // Loop through the atom_ids and populate the radius_values vector
    for (size_t i = 0; i < atom_ids.size(); ++i)
    {
        int atom_id = atom_ids[i];

        // Look up the radius corresponding to the atom_id
        auto it = value.find(atom_id);
        if (it != value.end())
        {
            _value[i] = it->second;
        }
        else
        {
            // Throw an exception if atom_id does not exist in the radii map
            throw std::runtime_error("Value for atom_id " +
                                     std::to_string(atom_id) + " not found.");
        }
    }

    return _value;
}

#endif
