"""dataRead.py"""
import os
import gzip
import numpy as np
# import pyvista as pv
from pmmoto.io import io_utils


__all__ = [
    "read_sphere_pack_xyzr_domain",
    # "read_vtk_grid",
    "read_r_lookup_file",
    "read_lammps_atoms",
    "read_atom_map",
    "read_rdf"
    ]

def read_sphere_pack_xyzr_domain(input_file):
    """Read in sphere pack given in x,y,z,(r)adius order including domain size

        Input File Format:
            x_min x_max
            y_min y_max
            z_min z_max
            x1 y1 z1 r1
            x2 y2 z2 r2
            x3 y3 z3 r3
    """

    # Check input file and proceed of exists
    io_utils.check_file(input_file)

    domain_file = open(input_file,'r',encoding="utf-8")
    lines = domain_file.readlines()
    num_spheres = len(lines) - 3

    sphere_data = np.zeros([num_spheres,4],dtype = np.double)
    domain_data = np.zeros([3,2],dtype = np.double)

    count_sphere = 0
    for n_line,line in enumerate(lines):
        if n_line < 3: # Grab domain size
            domain_data[n_line,0] =  float(line.split(" ")[0])
            domain_data[n_line,1] =  float(line.split(" ")[1])
        else: # Grab sphere
            try:
                for n in range(0,4):
                    sphere_data[count_sphere,n] = float(line.split(" ")[n])
            except ValueError:
                for n in range(0,4):
                    sphere_data[count_sphere,n] = float(line.split("\t")[n])
            count_sphere += 1

    domain_file.close()

    return sphere_data,domain_data

def read_vtk_grid(rank,size,file):
    """
    Read in parallel vtk file. Size must equal size when file written
    """
    # proc_files = os.listdir(file)
    # proc_files.sort()

    # # Check num_files is equal to mpi.size
    # io_utils.check_num_files(len(proc_files),size)

    # p_file = file + '/' + proc_files[rank]
    # data = pv.read(p_file)
    # array_name = data.array_names
    # grid_data = np.reshape(data[array_name[0]],data.dimensions,'F')

    # return np.ascontiguousarray(grid_data)

def read_r_lookup_file(input_file,power = 1):
    """
    Read in the radius lookup file for lammps simulations

    Actually reading in sigma

    File is:
    Atom_ID epsilon sigma

    """
    io_utils.check_file(input_file)

    r_lookup_file = open(input_file,'r',encoding="utf-8")

    sigma = {}  # Lennard-Jones
    lookup_lines = r_lookup_file.readlines()
  
    for n_line,line in enumerate(lookup_lines):
        sigma_i = float(line.split(" ")[2])
        sigma[n_line+1] = power*sigma_i
        
    r_lookup_file.close()

    return sigma

def read_lammps_atoms(input_file):
    """
        For the purposes of being able to specify arbitrarily
        sized particles, this definition of cutoff location
        relies on the assumption that the 'test particle'
        (as LJ interactions occur only between > 1 particles)
        has a LJsigma of 0. i.e. this evaluates the maximum possible
        size of the free volume network

        The following selects the energy minimum, i.e.
        movement of a particles center closer than this requires
        input force

        Input files has the following formats:
            XXXXXXXX
    """

    io_utils.check_file(input_file)

    if input_file.endswith('.gz'):
        domain_file = gzip.open(input_file,'rt')
    else:
        domain_file = open(input_file,'r',encoding="utf-8")

    lines = domain_file.readlines()
    domain_data = np.zeros([3,2],dtype = np.double)
    count_atom = 0
    for n_line,line in enumerate(lines):
        if n_line == 1:
            time_step = float(line)
        elif n_line == 3:
            num_objects = int(line)
            atom_data = np.zeros([num_objects, 3],dtype = np.double)
            atom_type = np.zeros(num_objects,dtype = int)
        elif 5 <= n_line <= 7:
            domain_data[n_line - 5,0] =  float(line.split(" ")[0])
            domain_data[n_line - 5,1] =  float(line.split(" ")[1])
        elif n_line >= 9:
            split = line.split(" ")
            atom_type[count_atom] = int(split[2])
            for count,n in enumerate([5,6,7]):
                atom_data[count_atom,count] = float(split[n]) # x,y,z,atom_id
            
            count_atom += 1

    # for n in range(0,num_objects):
    #     atom_ID = int(sphere_data[n,3])
    #     sphere_data[n,3] = r_lookup[atom_ID]

    domain_file.close()

    return atom_data,atom_type,domain_data

def read_rdf(input_folder):
    """
    Read input folder containing Radial Distribtuin Function Data 
    Folder must contain file called `atom_map.txt` and files for all 
    listed atoms of name 'atom_name'.rdf
    """

    # Check folder exists
    io_utils.check_folder(input_folder)

    # Check for atom_map.txt
    atom_map_file = input_folder + 'atom_map.txt'
    io_utils.check_file(atom_map_file)

    atom_map = read_atom_map(atom_map_file)

    # Check rdf files found for all atoms
    atom_files = []
    for atom in atom_map:
        atom_file = input_folder + atom + '.rdf'
        io_utils.check_file(atom_file)
        atom_files.append(atom_file)

    return atom_map,atom_files
        

def read_atom_map(input_file):
    """
    Read in the atom mapping file which has the following format:
        Atom_ID Atom_Name
    """
    # Check input file and proceed of exists
    io_utils.check_file(input_file)
    
    atom_file = open(input_file,'r',encoding="utf-8")
    atom_data= {}
    
    lines = atom_file.readlines()
    for line in lines:
        split = line.split(" ")
        label = split[1].split("\n")[0]
        ID = split[0]
        atom_data[label] = ID
    
    return atom_data