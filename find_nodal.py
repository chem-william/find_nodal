<<<<<<< HEAD
import argparse

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


ALIGN = False
=======
import utilities
import export_jmol
import argparse

import shutil
import os
import warnings

from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import seaborn as sns
from PIL import Image
from tqdm import tqdm
sns.set_style(style='white')


def dump_data(
    file,
    folder_name,
    planes,
    jmol_planes,
    max_pos_iso,
    max_neg_iso,
    angles,
    x_cross,
    y_cross,
    atom_info,
    xyz_vec,
    fitted_x,
    fitted_y,
    phis,
    max_iso_val,
    min_iso_val,
):

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    np.save(folder_name + "/planes.npy", planes)
    np.save(folder_name + "/jmol_plane_data.npy", jmol_planes)
    np.save(folder_name + "/max_pos_iso.npy", max_pos_iso)
    np.save(folder_name + "/max_neg_iso.npy", max_neg_iso)
    np.save(folder_name + "/angles.npy", angles)
    np.save(folder_name + "/x_cross.npy", x_cross)
    np.save(folder_name + "/y_cross.npy", y_cross)
    np.save(folder_name + "/atom_info", atom_info)
    np.save(folder_name + "/xyz_vec.npy", xyz_vec)
    np.save(folder_name + "/fitted_x.npy", fitted_x)
    np.save(folder_name + "/fitted_y.npy", fitted_y)
    np.save(folder_name + "/phis.npy", phis)
    np.save(folder_name + "/min_iso_val.npy", min_iso_val)
    np.save(folder_name + "/max_iso.npy", max_iso_val)

    shutil.copy(file, folder_name + "/" + file.split("/")[-1])
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc


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
<<<<<<< HEAD
            warnings.warn("Warning: y-component = 0. Might result in wrongly assigned direction leading to wrong angles.", RuntimeWarning)
=======
            warnings.warn(
                "Warning: y-component = 0. Might result in wrongly assigned direction leading to wrong angles.", RuntimeWarning
            )
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
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
    return datapoints - center


def find_nearest_idx(array, value: float) -> int:
    print((np.abs(array - value)))
    return (np.abs(array - value)).argmin()


def _p(x: np.array, *coeffs: np.array) -> np.array:
    return coeffs*x


def _check_bad_fit(r_val: float, p_val: float, point_count: int) -> None:
    # If the fit of this particular point is
    # very poor, warn the user. Thresholds are arbitrary
    if (np.abs(r_val) < 0.5) and p_val > 0.5:
<<<<<<< HEAD
        warnings.warn(f'R^2 = {r_val} and p-value = {p_val} for point #{point_count}. This fit might be very poor')
=======
        warnings.warn(
            f'R^2 = {r_val} and p-value = {p_val} for point #{point_count}. This fit might be very poor',
            RuntimeWarning
        )
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc


def draw_prev_angle(vec, angles, ax):
    # Draw angle
    angle_phi = np.linspace(0, -np.deg2rad(angles[-1]))
    angle_rho = 1.8
    _, line_phi = cart2pol(vec[0], vec[1])
    angle_phi += line_phi
    angle_x, angle_y = pol2cart(angle_rho, angle_phi)

    ax.plot(angle_x, angle_y, color='purple')


<<<<<<< HEAD
def fit_points(x, y, point_count: int) -> (np.array, np.array):
=======
def fit_points(x, y, point_count: int):
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
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

<<<<<<< HEAD
        slope, _, r_value, p_value, _ = stats.linregress(x_cart_rot,
                                                         y_cart_rot)
=======
        slope, _, r_value, p_value, _ = stats.linregress(
            x_cart_rot,
            y_cart_rot,
        )
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
        _check_bad_fit(r_value, p_value, point_count)

        xp = np.linspace(-1, 1, len(y_cart_rot))
        y_fit = _p(xp, slope)

        # Rotate fitted points to original position
        fit_rho, fit_phi = cart2pol(xp, y_fit)
        fit_phi += rot

        x, y = pol2cart(fit_rho, fit_phi)

    return x, y


<<<<<<< HEAD
def _string2vector(v) -> np.ndarray:
    # Enables inputs such as ('z'), ('-x'), etc.

    if isinstance(v, str):
        if v[0] == '-':
            return -_string2vector(v[1:])

        w = np.zeros(3)
        w['xyz'.index(v)] = 1.0

        return w

    return np.array(v, float)


def find_zero_crossings(plane: np.array, axis: int, x_cross, y_cross):
    u, indices = np.unique(plane[:, axis], return_index=True)
=======
def find_zero_crossings(plane, axis: int, x_cross, y_cross):
    _, indices = np.unique(plane[:, axis], return_index=True)
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc

    for value in plane[indices, axis]:
        line_indices = np.where(plane[:, axis] == value)
        temp_line = plane[line_indices]

        zero_crossings = np.where(np.diff(np.sign(temp_line[:, 3])))[0]

        for item in zero_crossings:
            x_cross.append((temp_line[item, 0] + temp_line[item + 1, 0])/2)
<<<<<<< HEAD
            y_cross.append((temp_line[item, 1] + temp_line[item + 1, 1]/2))


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


def plot_slices(ax,
                high_iso,
                low_iso,
                x_cross,
                y_cross,
                perp_vec,
                vec,
                angles,
                x,
                y):

    ax.plot([0, low_iso[0] - high_iso[0]],
            [0, low_iso[1] - high_iso[1]], color='orange')

    # TODO: check om navnene til de her punkter er rigtige
=======
            y_cross.append((temp_line[item, 1] + temp_line[item + 1, 1])/2)


def plot_slices(
    ax,
    high_iso,
    low_iso,
    x_cross,
    y_cross,
    perp_vec,
    vec,
    angles,
    x,
    y,
):

    ax.plot(
        [0, low_iso[0] - high_iso[0]],
        [0, low_iso[1] - high_iso[1]],
        color='orange'
    )

>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
    ax.scatter(high_iso[0], high_iso[1], marker='x', color='green')
    ax.scatter(low_iso[0], low_iso[1], marker='x', color='black')

    ax.plot([0, perp_vec[0]], [0, perp_vec[1]], color='black')

    # plot the midway points at which the
    # isovalues goes from positive to negative
<<<<<<< HEAD
    ax.scatter(x_cross, y_cross,
               marker='v', color='purple')

    # TODO: check hvad den her plotter
=======
    ax.scatter(x_cross, y_cross, marker='v', color='purple')

    # which direction the fitted line points to
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
    ax.scatter(vec[0], vec[1], color='black', marker='x')

    draw_prev_angle(vec, angles, ax)

    ax.plot(x, y, color='black', label=f'{angles[-1]}')
    ax.legend()


<<<<<<< HEAD
def main():
    file, ALIGN, show_slices = parse_args()

    # file = os.path.expanduser('~/Desktop') + '/helicity/pp/mo/56.cube'
    # file = os.getcwd() + "/helical.cube"
    file = os.getcwd() + "/cum_helical.cube"
    # file = os.getcwd() + "/helical_4cum.cube"

    atoms, all_info, xyz_vec = utilities.read_cube(file)

    if ALIGN:
        # Align data and atoms along z-axis
        # only necessary for .cube files from cubegen
        align_axis = atoms[0].position - atoms[4].position
        atoms.rotate(align_axis, 'z')

        all_info[:, :3] = _rotate_points(all_info[:, :3],
                                         align_axis,
                                         'z')

    all_info = all_info[all_info[:, 2].argsort()]

    # Center of the molecule is chosen to be Ru
    # center_atom = atoms[3].position

    # for the [4]cumulene
    center_atom = atoms[3].position
=======
def export_as_gif(imgs, x, y, fig, ax, folder_name):
    # First export the slices as a gif
    frames = []
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(
        "slices.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=400,
        loop=0,
    )

    # Next, export the change in angles
    ax.axis("off")
    fig.set_size_inches(10, 6)

    def update(i):
        if i != 0:
            ax.plot(x[:i], y[:i], marker='o', color='black')
        return ax

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(len(x)),
        interval=400,
    )
    anim.save(folder_name + "/angles.gif", dpi=100, writer="imagemagick")

    # Remove the .png's
    for file in imgs:
        os.remove(file)

    print("Exported slices.gif and angles.gif!")

    
def generate_xy_shift(bottom_carbon, top_carbon, carbon_chain, planes, atoms):
    zaxis = np.linspace(bottom_carbon[2], top_carbon[2], len(planes))
    x_correction = [pos.position[0] for pos in atoms[carbon_chain]]
    y_correction = [pos.position[1] for pos in atoms[carbon_chain]]

    old_axis = np.linspace(bottom_carbon[2], top_carbon[2], len(carbon_chain))
    interp_x = np.interp(zaxis, old_axis, x_correction)
    interp_y = np.interp(zaxis, old_axis, y_correction)

    return interp_x, interp_y


def main():
    parser = argparse.ArgumentParser(
        description='Analyze helicity of .cube file'
    )

    parser.add_argument(
        'file', metavar=".cube_file", help='Path to .cube file to analyze'
    )

    parser.add_argument(
        'center',
        metavar='center_atom',
        help=(
            'The atom to center data points around. '
            'Can be obtained by opening the file in e.g. Avogadro. '
            'First atom is 1'
        ),
    )

    parser.add_argument(
        'bottom_atom',
        help='"bottom" atom of the linear carbon chain',
    )

    parser.add_argument(
        'top_atom',
        help='"top" atom of the linear carbon chain',
    )
    
    parser.add_argument(
        '--carbon-chain',
        help=(
            '<Required> indices of carbon atoms ',
            'that defines the helix chain. Example "5,1,2,3,13"',
        ),
        required=True
    )

    parser.add_argument(
        '--open-jmol',
        dest='open_jmol',
        default=False,
        action='store_true',
        help=(
            'Open the generated .spt file in jmol after analysis. '
            'Default is False'
        )
    )

    parser.add_argument(
        '--show-slices',
        default=False,
        dest='show_slices',
        action='store_true',
        help=(
            'Show each slice that has been generated from the .cube file. '
            'Default is False'
        ),
    )

    parser.add_argument(
        '--export-gif',
        default=False,
        dest='export_gif',
        action='store_true',
        help='Export the slices as a .gif. Default is False'
    )
    args = parser.parse_args()

    file = os.getcwd() + '/' + args.file
    center_idx = int(args.center) - 1
    open_jmol = args.open_jmol
    show_slices = args.show_slices
    export_gif = args.export_gif
    carbon_chain = [int(item) for item in str(args.carbon_chain).split(",")]

    atoms, all_info, xyz_vec = utilities.read_cube(file)
    center_atom = atoms[int(center_idx)].position

    # carbons that define the start and end of the linear carbon chain.
    # Used to narrow the amount of slices along the z-axis
    # as we don't care about the orbitals on the substituents etc.
    top_carbon = atoms[int(args.top_atom) - 1].position
    bottom_carbon = atoms[int(args.bottom_atom) - 1].position
    if top_carbon[2] < bottom_carbon[2]:
        raise ValueError(
            "Top atom is lower on the z-axis than the bottom atom"
        )

>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
    center_x = center_atom[0]
    center_y = center_atom[1]
    center_z = center_atom[2]

    all_info[:, :3] = center_data(all_info[:, :3], center_atom)
    atoms = center_atoms(atoms, center_atom)

<<<<<<< HEAD
    # Lowest and highest carbon atom in terms of z-value
    # carbon1 = atoms[23]
    # carbon2 = atoms[6]

    # for the [4]cumulene
    carbon1 = atoms[5]
    carbon2 = atoms[1]

    planes = []
    plane = []
    print('Finding planes..')
    if ALIGN:
        pass
    else:
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
=======
    atom_info = [
        atoms, top_carbon, bottom_carbon, [center_x, center_y, center_z]
    ]

    planes = []
    plane = []
    # sorting is done so we can slice in planes along the z-axis
    all_info = all_info[all_info[:, 2].argsort()]

    prev_coord = all_info[0]
    for coordinate in tqdm(all_info, desc="Finding planes.."):
        if coordinate[2] == prev_coord[2]:
            # we're in the same plane so add the coordinate
            plane.append(
                [coordinate[0], coordinate[1], coordinate[2], coordinate[3]]
            )
        else:
            plane = np.array(plane)
            # Drop coordinates with isovalues == 0.0
            plane = plane[np.where(plane[:, 3] != 0.0)]

            if plane.size != 0:
                planes.append(plane)

            plane = []

        prev_coord = coordinate
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc

    planes = np.array(planes)

    pos_iso, neg_iso, plane_data = [], [], []
<<<<<<< HEAD
=======
    max_iso_val, min_iso_val = [], []
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
    angles = []
    prev_vec = None
    count = 1
    z_value = []
<<<<<<< HEAD

    print('Analyzing..')
    for idx, plane in enumerate(planes):
        if carbon2.position[2] > plane[0, 2] > carbon1.position[2]:
            if idx < len(planes) - 1:
                z_value.append(plane[0, 2])

                if show_slices:
                    fig, ax = plt.subplots()
                    color = ['red' if x < 0 else 'blue' for x in plane[:, 3]]
                    ax.scatter(plane[:, 0],
                               plane[:, 1],
                               s=abs(plane[:, 3])*800,
                               c=color,
                               alpha=0.6)

                # Find highest and lowest isovalue
                maximum = np.amax(plane[:, 3])
                max_index = np.where(plane[:, 3] == maximum)
                minimum = np.amin(plane[:, 3])
                min_index = np.where(plane[:, 3] == minimum)
                max_iso = [plane[max_index, 0], plane[max_index, 1]]
                max_iso = np.array(max_iso).ravel()
                min_iso = [plane[min_index, 0], plane[min_index, 1]]
                min_iso = np.array(min_iso).ravel()

                # Constrict number of points by a radius-filter
                r_indices = np.where(cart2pol(plane[:, 0], plane[:, 1])[0] < 1.3)
                plane = plane[r_indices]

                perp_vec = perpendicular_vector([max_iso[0] - min_iso[0], max_iso[1] - min_iso[1], 0])
=======
    imgs = []
    x_cross_collect = []
    y_cross_collect = []
    fitted_x = []
    fitted_y = []
    phis = []
    
    shift_x, shift_y = generate_xy_shift(
        bottom_carbon, top_carbon, carbon_chain, planes, atoms
    )


    print('Analyzing..')
    for idx, plane in enumerate(planes):
        if top_carbon[2] > plane[0, 2] > bottom_carbon[2]:
            if idx < len(planes) - 1:
                z_value.append(plane[0, 2])
                
                # Impose a shift on the coordinates to center the nodal plane
                plane[:, 0] -= shift_x[idx]
                plane[:, 1] -= shift_y[idx]

                if show_slices or export_gif:
                    _, ax = plt.subplots()
                    color = ['red' if x < 0 else 'blue' for x in plane[:, 3]]
                    plt.scatter(
                        plane[:, 0],
                        plane[:, 1],
                        s=abs(plane[:, 3])*800,
                        c=color,
                        alpha=0.6
                    )

                # Constrict number of points by a radius-filter
                r_indices = np.where(
                    cart2pol(plane[:, 0], plane[:, 1])[0] < 1.
                )
                plane = plane[r_indices]

                # Find highest and lowest isovalue
                maximum = np.amax(plane[:, 3])
                max_iso_val.append(maximum)
                max_index = np.where(plane[:, 3] == maximum)

                minimum = np.amin(plane[:, 3])
                min_iso_val.append(minimum)
                min_index = np.where(plane[:, 3] == minimum)

                max_iso = [plane[max_index, 0], plane[max_index, 1]]
                max_iso = np.array(max_iso).ravel()

                phis.append(cart2pol(max_iso[0], max_iso[1])[1])

                min_iso = [plane[min_index, 0], plane[min_index, 1]]
                min_iso = np.array(min_iso).ravel()
                
                plane = plane[np.lexsort((plane[:, 0], plane[:, 1]))]

                perp_vec = perpendicular_vector(
                    [max_iso[0] - min_iso[0], max_iso[1] - min_iso[1], 0])
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
                perp_vec = perp_vec/np.linalg.norm(perp_vec)

                # Sort plane - first by x- then by y-values
                plane = plane[np.lexsort((plane[:, 0], plane[:, 1]))]

                # Find each line of the plot + zero crossings
                x_cross = []
                y_cross = []
                find_zero_crossings(plane, 0, x_cross, y_cross)
                find_zero_crossings(plane, 1, x_cross, y_cross)
<<<<<<< HEAD

                x, y = fit_points(x_cross, y_cross, count)
=======
                x_cross_collect.append(x_cross)
                y_cross_collect.append(y_cross)

                x, y = fit_points(x_cross, y_cross, count)
                fitted_x.append(x)
                fitted_y.append(y)
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc

                def unit_vector(vector):
                    return vector / np.linalg.norm(vector)

                def angle_between(v1, v2):
                    v1_u = unit_vector(v1)
                    v2_u = unit_vector(v2)
<<<<<<< HEAD
                    rad = np.arctan2(np.linalg.det([v1_u, v2_u]), np.dot(v1_u, v2_u))
=======
                    rad = np.arctan2(np.linalg.det(
                        [v1_u, v2_u]), np.dot(v1_u, v2_u))
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc

                    return np.rad2deg(rad)

                vec = [x[-1], y[-1]]
                if np.dot(perp_vec, vec) < 0:
                    vec = [x[0], y[0]]
                else:
                    vec = [x[-1], y[-1]]

                if count == 1:
                    angles.append(0)
                else:
                    angle = angle_between(prev_vec, vec)
                    angles.append(angle)

<<<<<<< HEAD
                # ax.set_zlabel('Z axis')

                prev_vec = vec
                count += 1

                pos_iso.append([max_iso[ 0], max_iso[1], plane[0, 2]])
                neg_iso.append([min_iso[0], min_iso[1], plane[0, 2]])
                plane_data.append([vec[0], vec[1], plane[0, 2]])

                if show_slices:
                    limits = 6
                    plot_slices(ax,
                                max_iso,
                                min_iso,
                                x_cross,
                                y_cross,
                                perp_vec,
                                vec,
                                angles,
                                x,
                                y)
=======
                prev_vec = vec
                count += 1

                pos_iso.append([max_iso[0], max_iso[1], plane[0, 2]])
                neg_iso.append([min_iso[0], min_iso[1], plane[0, 2]])
                plane_data.append([vec[0], vec[1], plane[0, 2]])

                if show_slices or export_gif:
                    limits = 6
                    plot_slices(
                        ax,
                        max_iso,
                        min_iso,
                        x_cross,
                        y_cross,
                        perp_vec,
                        vec,
                        angles,
                        x,
                        y,
                    )
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
                    ax.set_xlim([-limits, limits])
                    ax.set_ylim([-limits, limits])

                    ax.set_xlabel('X axis')
                    ax.set_ylabel('Y axis')

<<<<<<< HEAD

    data_dic = {'pos_iso': pos_iso,
                'neg_iso': neg_iso,
                'plane_data': plane_data}

=======
                    if export_gif and idx % 2 == 0:
                        file_name = str(idx) + "_slice.png"
                        plt.savefig(file_name, bbox_inches="tight")
                        imgs.append(file_name)

    data_dic = {
        'pos_iso': pos_iso,
        'neg_iso': neg_iso,
        'plane_data': plane_data
    }

    # new figure to plot the changes in angle
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc
    fig, ax = plt.subplots()

    # Overlay atoms on angle-plot
    y_value = np.sum(angles)/2
    for idx, atom in enumerate(atoms):
        if atom.symbol == 'C':
<<<<<<< HEAD
            ax.scatter(atom.position[2] + carbon2.position[2], y_value, color='black', marker='o', s=10)
            ax.annotate(idx, (atom.position[2] + carbon2.position[2], y_value))

        if atom.symbol == 'Ru':
            ax.scatter(atom.position[2] + carbon2.position[2], y_value, color='turquoise', marker='o', s=10)

    ax.plot(np.array(z_value) + carbon2.position[2], np.cumsum(angles), marker='o')
    ax.plot(np.array(z_value) + carbon2.position[2], np.gradient(np.cumsum(angles)), marker='o')

    export_jmol.export_jmol(data_dic, [center_x, center_y, center_z], file)
    
    # Automatically run jmol - i'm lazy
    cmd = 'jmol jmol_export.spt &'
    os.system(cmd)

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze helicity of .cube file')

    parser.add_argument('file', metavar="path to .cube file",
                        help='path to .cube file to analyze')

    parser.add_argument('center', metavar='center_atom'
                        'the atom to center data points around. Can be obtained by opening the file in e.g. Avogadro. First atom is 1')

    parser.add_argument('--no-align',
                        dest='no_align',
                        default=True,
                        action='store_false',
                        help='whether the datapoints should be aligned to z-axis. Should be False for .cube files from transmission calculations. Default is True')

    parser.add_argument('--open-jmol',
                        default=False,
                        help='whether to open the generated .spt file in jmol after analysis. Default is False')

    parser.add_argument('--show-slices',
                        default=False,
                        dest='show_slices',
                        action='store_true',
                        help='show each slice that has been generated from the .cube file. Default is False')
    args = parser.parse_args()

    file = os.getcwd() + '/' + args.file
    center_atom = int(args.center) - 1
    open_jmol = args.open_jmol


    return file, args.no_align, args.show_slices
=======
            ax.scatter(
                atom.position[2] + bottom_carbon[2],
                y_value,
                color='black',
                marker='o',
                s=10
            )
            ax.annotate(idx, (atom.position[2] + bottom_carbon[2], y_value))

        if atom.symbol == 'Ru':
            ax.scatter(
                atom.position[2] + bottom_carbon[2],
                y_value,
                color='turquoise',
                marker='o',
                s=10
            )

    x = np.array(z_value) + bottom_carbon[2]
    y = np.cumsum(angles)

    ax.set_xlim([x[0] - 1, x[-1] + 1])
    ax.set_ylim([y[0], y[-1]])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    ax.plot(x, y, marker='o', color='blue')
    ax.plot(x, np.gradient(y), marker='o')

    names = file.split("/")
    folder_name = names[-3] + "_" + names[-1][:-5]
    
    if open_jmol:
        cmd = 'jmol ' + folder_name + '/jmol_export.spt &'
        os.system(cmd)

    # show all figures at the same time
    if show_slices:
        plt.show()

    # Save data
    dump_data(
        file,
        folder_name,
        planes,
        data_dic["plane_data"],
        data_dic["pos_iso"],
        data_dic["neg_iso"],
        angles,
        x_cross_collect,
        y_cross_collect,
        atom_info,
        xyz_vec,
        fitted_x,
        fitted_y,
        phis,
        max_iso_val,
        min_iso_val,
    )

    if export_gif:
        export_as_gif(imgs, x, y, fig, ax, folder_name)

    export_jmol.export_jmol(
            data_dic, [center_x, center_y, center_z], file, folder_name
        )

    plt.savefig(folder_name + "/angles.png", bbox_inches="tight")
>>>>>>> c6109251cfcd5061f338d8088cb5f9703b22ddbc


if __name__ == '__main__':
    exit(main())
