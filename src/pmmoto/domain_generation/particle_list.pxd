"""particles.pxd"""

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

cdef extern from "particles.hpp":

    cdef cppclass Particle

	cdef cppclass ParticleList:
		vector[Particle] atoms
		AtomList(
			const vector[vector[double]]& atom_data, 
			bool build_kd
			)
		size_t size()

	cdef unique_ptr[ParticleList] initialize_list(
		vector[vector[double]] data,
		vector[double] point,
		double radius,
		bool kd_tree,
		bool trim
	)