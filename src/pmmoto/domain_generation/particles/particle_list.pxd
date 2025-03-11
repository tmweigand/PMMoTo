"""particles.pxd"""

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef extern from "particle_list.hpp":

	cdef struct Box:
		double[3] min,max,length

	cdef cppclass Particle

	cdef cppclass ParticleList:
		ParticleList(vector[vector[double]] particle_data)

	cdef shared_ptr[ParticleList] initialize_particles(
		vector[vector[double]] data,
		vector[vector[double]] domain_box,
		vector[vector[double]] subdomain_box,
		bool add_periodic
	)

	cdef vector[vector[double]] return_particles(
		shared_ptr[ParticleList]
	)