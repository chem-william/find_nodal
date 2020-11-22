import os
import numpy as np
from ase.units import Bohr
from ase.atoms import Atoms
<<<<<<< HEAD
=======
from tqdm import tqdm
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
import libread_cube


def _neg_atoms(line) -> bool:
    if int(line[0]) < 0:
        return True
    else:
        return False


def read_cube(file):
    """
    Can't handle .CUBE files with more than 1 dataset in it

    Returns a dict with the following keys:

    * 'atoms': Atoms object
    * 'all_info' : (x, y, z, iso_value) ndarray
    * 'xyz_vec': unit vector in the x, y and z direction
    """
    xyz_vec = [0, 0, 0]

    print('Loading cube file..')
    print('File: {}'.format(file))
    with open(file) as fp:
        # Discard the first and second line - just a comment
        fp.readline()
        fp.readline()

        # Third line is number of atoms - this value can be negative
        line = fp.readline().split()
        neg_atoms = _neg_atoms(line)

        natoms = abs(int(line[0]))
        end_index = None
        if neg_atoms:
            end_index = -1

        origin = np.array([float(x) * Bohr for x in line[1:end_index]])
        cell = np.empty((3, 3))
        shape = []

        # Next three lines contain the cell information
        for i in range(3):
            line = fp.readline().split()

            n, x, y, z = [float(s) for s in line]
            shape.append(int(n))

            # Get the unit vectors of each dimension
            xyz_vec[i] = float(line[i + 1])

            # Size of the cell where all the voxels are
            cell[i] = n * Bohr * np.array([x, y, z])

        # Get the information about the atoms
        numbers = np.empty(natoms, int)
        positions = np.empty((natoms, 3))
        for i in range(natoms):
            line = fp.readline().split()
            numbers[i] = int(line[0])
            positions[i] = [float(s) for s in line[2:]]
        positions *= Bohr

        # Transform the info about the atoms to an ase.Atoms object
        # and align along z-axis
        atoms = Atoms(numbers=numbers, positions=positions, cell=cell)
        dct = {'atoms': atoms}

        # It can not handle multiple datasets in the same .cube file
        if neg_atoms:
            amount_datasets = int(fp.readline().split()[0])
            if amount_datasets > 1:
                raise ValueError(f'{file} contains more than 1 dataset '
                                 f'({amount_datasets} datasets)')

        # Read the voxel information
        cube_data = []
        for line in fp:
            for item in line.split():
                cube_data.append(float(item))

        cube_data = np.array(cube_data)

        # all_info = _get_info(cube_data, xyz_vec, shape)*Bohr
        all_info = np.array(
                libread_cube.get_info(cube_data, xyz_vec, shape)
            )
        all_info *= Bohr

        # Center points around origin
        all_info[:, :3] += origin

        dct['all_info'] = all_info

        return dct['atoms'], dct['all_info'], xyz_vec


def _get_info(data, xyz_vec, dims):
    all_info = []
    total = 0

<<<<<<< HEAD
    for ix in range(dims[0]):
=======
    for ix in tqdm(range(dims[0]), desc="Getting coordinates from voxels.."):
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
        x = ix*xyz_vec[0]

        for iy in range(dims[1]):
            y = iy*xyz_vec[1]

            for iz in range(dims[2]):
                z = iz*xyz_vec[2]

                all_info.append([x, y, z, data[total]])

                total += 1

    all_info = np.array(all_info)

    return all_info


def main():
    # file = os.getcwd() + "/helical.cube"
    file = os.getcwd() + "/helical_4cum.cube"
    # file = "/home/william/Desktop/helical.cube"

    atoms, all_info, xyz_vec = read_cube(file)


if __name__ == '__main__':
    exit(main())
