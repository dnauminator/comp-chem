import argparse
import numpy
import os

def calc_dist(atom1_coord, atom2_coord):
    """Calculates the distance between points in 3d space."""
    x_dist = atom1_coord[0]-atom2_coord[0]
    y_dist = atom1_coord[1]-atom2_coord[1]
    z_dist = atom1_coord[2]-atom2_coord[2]
    bond_len_12 = numpy.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
    return bond_len_12

def bond_check(atom_distance, min_length=0, max_length=1.5):
    """Checks if a distance is a bond based on minimum and maximum bond lengths."""
    if atom_distance > min_length and atom_distance <= max_length:
        return True
    else:
        return False

def open_xyz(xyzfilename):
    """Opens, reads xyz file. Outputs a tuple of symbols and coordinates."""
    xyz_file = numpy.genfromtxt(fname = xyzfilename, skip_header = 2, dtype = 'unicode')
    symbols = xyz_file[:,0]
    coord = (xyz_file[:,1:])
    coord = coord.astype(numpy.single)
    return symbols, coord

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This script analyzes a user given xyz file and outputs the length of the bonds.")
    parser.add_argument("xyz_file", help="The filepath for the xyz file to analyze.")
    # below lines are an optional arguments that tie to bond_check function
    parser.add_argument('-min_length', help='The minimum distance to consider atoms bonded.', type=float, default=0)
    parser.add_argument('-max_length', help='The minimum distance to consider atoms bonded.', type=float, default=1.5)
    
    args = parser.parse_args()

    xyzfilename = args.xyz_file

    symbols, coord = open_xyz(xyzfilename) # this must go after open_xyz function definition

    # file_location = os.path.join('data', 'water.xyz')
    # symbols, coordinates = open_xyz(file_location)
    # above commented out as we are using argparse to get the file location
    num_atoms = len(symbols)
    for num1 in range(0, num_atoms):
        for num2 in range(0, num_atoms):
            if num1 < num2:
                bond_len_12 = calc_dist(coord[num1], coord[num2])
                if bond_check(bond_len_12, min_length=args.min_length, max_length=args.max_length) is True:
                    print(F'{symbols[num1]} to {symbols[num2]} : {bond_len_12:.3f}')