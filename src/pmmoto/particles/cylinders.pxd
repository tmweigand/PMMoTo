"""cylinders.pxd"""
from libcpp.vector cimport vector

cdef extern from "cylinders.hpp":

	cdef cppclass CylinderList:
		
		CylinderList(
			vector[vector[double]] point_1,
            vector[vector[double]] point_2,
			vector[double] radii,
		) except +