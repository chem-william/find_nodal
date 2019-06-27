from time import time
import os
import warnings

from mpl_toolkits.mplot3d import Axes3D # noqa
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style(style='white')

import export_jmol
import utilities


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return x, y


def perpendicular_vector(v: list) -> list:
    thres = 0.000005
    v = np.array(v)
    if v[1] != thres:
        v -= thres

    if v[1] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            warnings.warn("Warning: y-component = 0. Might result in wrongly assigned direction leading to wrong angles.", RuntimeWarning)
            return [0, -1]

    return [-v[1], v[0]]


def center_atoms(atoms, center_atom):
    x = center_atom[0]
    y = center_atom[1]
    z = center_atom[2]

    # Centering atoms around given atom
    for idx, atom in enumerate(atoms):
        atoms[idx].position[0] = atom.position[0] - x
        atoms[idx].position[1] = atom.position[1] - y
        atoms[idx].position[2] = atom.position[2] - z

    return atoms


def center_data(datapoints, center):
    for idx, point in enumerate(datapoints):
        datapoints[idx] = point - center

    return datapoints


def find_nearest_idx(array, value: float) -> int:
    print((np.abs(array - value)))
    return (np.abs(array - value)).argmin()


def _p(x: np.array, *coeffs: np.array) -> np.array:
    return coeffs*x


def _check_bad_fit(r_val: float, p_val: float, point_count: int) -> None:
    # If the fit of this particular point is
    # very poor, warn the user. Thresholds are arbitrary
    if (np.abs(r_val) < 0.5) and p_val > 0.5:
        warnings.warn(f'R^2 = {r_val} and p-value = {p_val} for point #{point_count}. This fit might be very poor')


def draw_prev_angle(vec, angles, ax):
    # Draw angle
    angle_phi = np.linspace(0, -np.deg2rad(angles[-1]))
    angle_rho = 1.8
    _, line_phi = cart2pol(vec[0], vec[1])
    angle_phi += line_phi
    angle_x, angle_y = pol2cart(angle_rho, angle_phi)
    ax.plot(angle_x, angle_y, color='purple')


def fit_points(x, y, point_count: int) -> (np.array, np.array):
    x = np.array(x)
    y = np.array(y)

    if np.var(x) > 0.03:
        xp = np.linspace(-1, 1, len(y))

        slope, _, r_value, p_value, _ = stats.linregress(x, y)
        _check_bad_fit(r_value, p_value, point_count)

        x, y = xp, _p(xp, slope)

    else:
        # Rotate points if most has x = 0
        rot = np.deg2rad(45)*len(x)  # 45 degrees in radians
        rho, phi = cart2pol(x, y)
        phi -= rot
        x_cart_rot, y_cart_rot = pol2cart(rho, phi)

        slope, _, r_value, p_value, _ = stats.linregress(x_cart_rot,
                                                         y_cart_rot)
        _check_bad_fit(r_value, p_value, point_count)

        xp = np.linspace(-1, 1, len(y_cart_rot))
        y_fit = _p(xp, slope)

        # Rotate fitted points to original position
        fit_rho, fit_phi = cart2pol(xp, y_fit)
        fit_phi += rot

        x, y = pol2cart(fit_rho, fit_phi)

    return x, y


def _string2vector(v) -> np.ndarray:
    # Enables inputs such as ('z'), ('-x'), etc.

    if isinstance(v, str):
        if v[0] == '-':
            return -_string2vector(v[1:])

        w = np.zeros(3)
        w['xyz'.index(v)] = 1.0

        return w

    return np.array(v, float)


def _unit_vec(vec) -> np.array:
    vec = _string2vector(vec)
    norm_vec = np.linalg.norm(vec)
    vec /= norm_vec

    if norm_vec == 0.0:
        raise ZeroDivisionError('Cannot rotate: norm(v) == 0')

    return vec


def _rotate_points(points, a, v=None):
    """Rotate points based on two vectors.

    Parameters:

    a = None:
        Angle that the points is rotated around the vecor 'v'. 'a'
        can also be a vector and then 'a' is rotated
        into 'v'.

    v:
        Vector to rotate the atoms around. Vectors can be given as
        strings: 'x', '-x', 'y', ... .

    center = (0, 0, 0):
        The center is kept fixed under the rotation.

    Examples:

    Rotate 90 degrees around the z-axis, so that the x-axis is
    rotated into the y-axis:

    >>> atoms.rotate('x', 'y')
    >>> atoms.rotate((1, 0, 0), (0, 1, 0))
    """

    assert a is not None

    v = _unit_vec(v)

    v2 = _unit_vec(a)

    c = np.dot(v, v2)
    v = np.cross(v, v2)
    s = np.linalg.norm(v)

    # In case 'v' and 'a' are parallel, np.cross(v, v2) vanish
    # and can't be used as a rotation axis. However, in this
    # case any rotation axis perpendicular to v2 will do.
    eps = 1e-7
    if s < eps:
        v = np.cross((0, 0, 1), v2)
        if np.linalg.norm(v) < eps:
            v = np.cross((1, 0, 0), v2)
        assert np.linalg.norm(v) >= eps

    result = (c * points
              - np.cross(points, s * v)
              + np.outer(np.dot(points, v), (1.0 - c) * v))

    return result


# XXX: virker ikke pt, og problemet er sandsynligvis, at punkterne ikke er centret
# hvorfor, det ved jeg ikke

def main():
    # file = os.path.expanduser('~/Desktop') + '/helicity/pp/mo/56.cube'
    # file = os.path.expanduser('~/Desktop') + '/4cum_helical.cube'
    file = os.getcwd() + "/helical.cube"
    # file = os.getcwd() + "/helical_4cum.cube"
    # file = os.getcwd() + "/h.cube"

    # fig, ax = plt.subplots()

    atoms, all_info, xyz_vec = utilities.read_cube(file)

    # Align data along z-axis
    # align_axis = atoms[0].position - atoms[4].position
    # atoms.rotate(align_axis, 'z')

    # all_info[:, :3] = _rotate_points(all_info[:, :3],
    #                                  align_axis,
    #                                  'z')

    all_info = all_info[all_info[:, 2].argsort()]

    # plt.plot(range(len(all_info[p1:p2, 2])), all_info[p1:p2, 2], "-o")
    # plt.show()


    # Center of the molecule is chosen to be Ru
    center_atom = atoms[3].position

    # for the [4]cumulene
    # center_atom = atoms[2].position
    center_x = center_atom[0]
    center_y = center_atom[1]
    center_z = center_atom[2]

    all_info[:, :3] = center_data(all_info[:, :3], atoms[3].position)
    atoms = center_atoms(atoms, center_atom)

    # ax = plt.axes(projection='3d')

    # step = 10
    # ax.scatter(all_info[::step, 0],
    #            all_info[::step, 1],
    #            all_info[::step, 2],
    #            s=all_info[::step, 3]*100)

    # for atom in atoms:
    #     if atom.symbol == 'C':
    #         ax.scatter(atom.position[0],
    #                    atom.position[1],
    #                    atom.position[2],
    #                    c='black')

    #     if atom.symbol == 'Ru':
    #         ax.scatter(atom.position[0],
    #                     atom.position[1],
    #                     atom.position[2],
    #                     c='turquoise')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')

    # plt.show()

    # Lowest and highest carbon atom in terms of z-value
    carbon1 = atoms[23]
    carbon2 = atoms[6]

    # for the [4]cumulene
    # carbon1 = atoms[4]
    # carbon2 = atoms[0]
    # prev_coord = 0
    # temp = []
    # for i, coordinate in enumerate(all_info):
    #     if i == 0:
    #         prev_coord = coordinate
    #     else:
    #         temp.append(coordinate[0] - prev_coord[0])
    #         prev_coord = coordinate
    # plt.plot(range(len(temp)), (temp))
    # plt.show()

    print('Finding planes..')
    planes = []
    plane = []
    prev_coord = all_info[0]
    for coordinate in all_info:
        # Don't bother with coordinates that has an isovalue of 0
        if not coordinate[3] == 0.0:
            if coordinate[2] == prev_coord[2]:
                # we're in the same plane so add the coordinate
                plane.append([coordinate[0],
                              coordinate[1],
                              coordinate[2],
                              coordinate[3]])
            else:
                plane = np.array(plane)
                planes.append(plane)
                plane = []

        prev_coord = coordinate

    planes = np.array(planes)

    pos_iso, neg_iso, plane_data = [], [], []
    angles = []
    prev_vec = None
    count = 1
    z_value = []

    ax = plt.axes(projection='3d')

    step = 1
    for idx, plane in enumerate(planes):
        color = ['red' if x < 0 else 'blue' for x in plane[::step, 3]]

        ax = plt.axes(projection='3d')
        ax.scatter(plane[::step, 0],
                   plane[::step, 1],
                   plane[::step, 2],
                   s=abs(plane[::step, 3])*800,
                   # c=color,
                   alpha=0.6)
    plt.show()

    print('Cleaning values..')
    for idx, plane in enumerate(planes):
        if carbon2.position[2] > plane[0, 2] > carbon1.position[2]:
            if idx < len(planes) - 1:
                z_value.append(plane[0, 2])
                fig, ax = plt.subplots()

                color = ['red' if x < 0 else 'blue' for x in plane[:, 3]]

                # ax = plt.axes(projection='3d')
                ax.scatter(plane[:, 0],
                           plane[:, 1],
                        #    plane[:, 2],
                           s=abs(plane[:, 3])*800,
                           c=color,
                           alpha=0.6)
                # plt.show()

                # Find highest and lowest isovalue
                maximum = np.amax(plane[:, 3])
                max_index = np.where(plane[:, 3] == maximum)
                minimum = np.amin(plane[:, 3])
                min_index = np.where(plane[:, 3] == minimum)
                p1 = [plane[max_index, 0], plane[max_index, 1]]
                p1 = np.array(p1).ravel()
                p2 = [plane[min_index, 0], plane[min_index, 1]]
                p2 = np.array(p2).ravel()

                # Constrict number of points by a radius-filter
                r_indices = np.where(cart2pol(plane[:, 0], plane[:, 1])[0] < 1.5)
                plane = plane[r_indices]

                plt.plot([0, p2[0] - p1[0]], [0, p2[1] - p1[1]], color='orange')

                plt.scatter(p1[0], p1[1], marker='x', color='green')
                plt.scatter(p2[0], p2[1], marker='x', color='black')

                perp_vec = perpendicular_vector([p1[0] - p2[0], p1[1] - p2[1], 0])
                perp_vec = perp_vec/np.linalg.norm(perp_vec)
                plt.plot([0, perp_vec[0]], [0, perp_vec[1]], color='black')

                # Sort plane - first by x- then by y-values
                plane = plane[np.lexsort((plane[:, 0], plane[:, 1]))]

                # Find each line of the plot + zero crossings
                x_cross = []
                y_cross = []
                u, indices = np.unique(plane[:, 1], return_index=True)
                for value in plane[indices, 1]:
                    line_indices = np.where(plane[:, 1] == value)
                    temp_line = plane[line_indices]

                    zero_crossings = np.where(np.diff(np.sign(temp_line[:, 3])))[0]

                    for item in zero_crossings:
                        x_cross.append((temp_line[item, 0] + temp_line[item + 1, 0])/2)
                        y_cross.append((temp_line[item, 1] + temp_line[item + 1, 1]/2))

                u, indices = np.unique(plane[:, 0], return_index=True)
                for value in plane[indices, 0]:
                    line_indices = np.where(plane[:, 0] == value)
                    temp_line = plane[line_indices]

                    zero_crossings = np.where(np.diff(np.sign(temp_line[:, 3])))[0]

                    for item in zero_crossings:
                        x_cross.append((temp_line[item, 0] + temp_line[item + 1, 0])/2)
                        y_cross.append((temp_line[item, 1] + temp_line[item + 1, 1]/2))

                ax.scatter(x_cross, y_cross,
                        #    plane[0, 2],
                           marker='v', color='purple')

                x, y = fit_points(x_cross, y_cross, count)

                def unit_vector(vector):
                    return vector / np.linalg.norm(vector)

                def angle_between(v1, v2):
                    v1_u = unit_vector(v1)
                    v2_u = unit_vector(v2)
                    rad = np.arctan2(np.linalg.det([v1_u, v2_u]), np.dot(v1_u, v2_u))

                    return np.rad2deg(rad)

                vec = [x[-1], y[-1]]
                if np.dot(perp_vec, vec) < 0:
                    vec = [x[0], y[0]]
                else:
                    vec = [x[-1], y[-1]]

                ax.scatter(vec[0], vec[1], color='black', marker='x')

                if count == 1:
                    angles.append(0)

                else:
                    angle = angle_between(prev_vec, vec)
                    angles.append(angle)

                draw_prev_angle(vec, angles, ax)

                ax.plot(x, y, color='black', label=f'{angles[-1]}')
                ax.legend()

                limits = 6
                ax.set_xlim([-limits, limits])
                ax.set_ylim([-limits, limits])

                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                # ax.set_zlabel('Z axis')

                prev_vec = vec
                count += 1

                pos_iso.append([p1[ 0], p1[1], plane[0, 2]])
                neg_iso.append([p2[0], p2[1], plane[0, 2]])
                plane_data.append([vec[0], vec[1], plane[0, 2]])

                plt.show()

    data_dic = {'pos_iso': pos_iso,
                'neg_iso': neg_iso,
                'plane_data': plane_data}
    fig, ax = plt.subplots()

    # Overlay atoms on angle-plot
    y_value = np.sum(angles)/2
    for idx, atom in enumerate(atoms):
        if atom.symbol == 'C':
            ax.scatter(atom.position[2] + carbon2.position[2], y_value, color='black', marker='o', s=10)
            ax.annotate(idx, (atom.position[2] + carbon2.position[2], y_value))

        if atom.symbol == 'Ru':
            ax.scatter(atom.position[2] + carbon2.position[2], y_value, color='turquoise', marker='o', s=10)

    ax.plot(np.array(z_value) + carbon2.position[2], np.cumsum(angles), marker='o')
    ax.plot(np.array(z_value) + carbon2.position[2], np.gradient(np.cumsum(angles)), marker='o')

    export_jmol.export_jmol(data_dic, [center_x, center_y, center_z])
    
    # Automatically run jmol - i'm lazy
    cmd = 'jmol jmol_export.spt &'
    os.system(cmd)
    plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(pval, rval)
    # for i in range(len(rval)):
    #     ax.annotate(i+1, (pval[i], rval[i]))


    # print('Plotting atoms..')
    # for atom in atoms:
    #     if atom.symbol == 'C':
    #         ax.scatter(atom.position[0],
    #                    atom.position[1],
    #                    atom.position[2],
    #                    c='black')

    #     if atom.symbol == 'Ru':
    #         ax.scatter(atom.position[0],
    #                    atom.position[1],
    #                    atom.position[2],
    #                    c='turquoise')

    # plt.show()


if __name__ == '__main__':
    exit(main())
