"""new_spheres.pxd"""

cdef extern from "new_spheres.cpp":
	pass

cdef extern from "new_spheres.hpp":
	void test()
	cdef cppclass NewSphereList:
		NewSphereListNewSphereList(double radii) except +
		NewSphereList() except +
