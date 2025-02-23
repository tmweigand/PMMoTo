#ifndef ATOMS_H
#define ATOMS_H

#include "particle_list.hpp"

/**
 * @class Atom
 * @brief Represents a sphere with position and radius.
 */
class Atom : public Particle {
public:
  Atom(double x, double y, double z, double radius)
      : Particle({x, y, z}, radius) {}
  Atom(std::vector<double> coordinates, double radius)
      : Particle(coordinates, radius) {}
  Atom(std::vector<double> atom_data) : Particle(atom_data, atom_data[3]) {}
};

class AtomList : public ParticleList<Atom> {

public:
  std::vector<Atom> atoms;
  AtomList(std::vector<Atom> atom_data, bool build_kd = false,
           double radius = 0.)
      : ParticleList<Atom>(atom_data) {
    for (const auto &atom : atom_data) {
      atoms.emplace_back(atom.coordinates[0], atom.coordinates[1],
                         atom.coordinates[2], radius);
    }
    if (build_kd) {
      std::vector<std::vector<double>> coords;
      for (const auto &atom : atom_data) {
        coords.push_back(
            {atom.coordinates[0], atom.coordinates[1], atom.coordinates[2]});
      }
      initializeKDTree(coords);
    }
  }

  void initialize_kd() {
    std::vector<std::vector<double>> coords;
    for (const auto &atom : atoms) {
      coords.push_back(
          {atom.coordinates[0], atom.coordinates[1], atom.coordinates[2]});
    }
    std::cout << "Coords length: " << coords.size() << "\n";
    initializeKDTree(coords);
  }

  size_t size() const { return atoms.size(); }

  std::vector<std::vector<double>> getAllCoordinates() const {
    std::vector<std::vector<double>> coords;
    for (const auto &atom : atoms) {
      coords.push_back(
          {atom.coordinates[0], atom.coordinates[1], atom.coordinates[2]});
    }
    return coords;
  }
};

#endif
