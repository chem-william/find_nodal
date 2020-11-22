import numpy.linalg as la
from typing import List
from typing import Dict

EXPORT_FILE = "jmol_export.spt"


def draw_iso(iso_values: List[List[int]],
             origin: List[int],
             file,
             color: str,
             idx: int) -> None:
    line_str = f'draw line{idx} LINE line width 0.2 color {color} '

    for value in iso_values:
        coord = [origin[0] + value[0],
                 origin[1] + value[1],
                 origin[2] + value[2]]
        line_str += f'{{{coord[0]} {coord[1]} {coord[2]}}} '
    file.write(line_str)
    file.write('\n')


def draw_curve(line_values, file):
    curve_str = f"draw curve1 CURVE curve width 0.3 color black"

    for value in line_values:
        curve_str += f"{{ {value[0]} {value[1]} {value[2]} }}"

    file.write(curve_str)
    file.write("\n")


def draw_plane(values, origin: List[int], file) -> None:
    prev_coord = None
    line_str: str = ''
    for idx, value in enumerate(values):
        norm_xy = [value[0], value[1]]
        norm_xy = norm_xy/la.norm(norm_xy)

        coord = [origin[0] + norm_xy[0],
                 origin[1] + norm_xy[1],
                 origin[2] + value[2]]

        if idx > 1:
            # Upper left
            line_str = (
                    f"draw plane{idx} PLANE "
                    f"{{{coord[0]} {coord[1]} {coord[2]}}} "
                )
            # Upper right
            line_str += (
                    f"{{{prev_coord[0]} {prev_coord[1]} {prev_coord[2]}}} "
                )
            # Lower right
            line_str += (
                    f"{{{coord[0] - norm_xy[0]*2} "
                    f"{coord[1] - norm_xy[1]*2} "
                    f"{prev_coord[2]}}} "
                )
            # line_str += f'draw circle{idx} DIAMETER 0.11 CIRCLE {{ {coord[0] - norm_xy[0]*2} {coord[1] - norm_xy[1]*2} {coord[2]} }} '
            # Lower left
            line_str += (
                    f"{{{prev_coord[0] - norm_xy[0]*2} "
                    f"{prev_coord[1] - norm_xy[1]*2} "
                    f"{coord[2]}}} "
                )

            file.write(line_str)
            file.write('\n')
            prev_coord = coord
        else:
            prev_coord = coord


def export_jmol(
        export_data: Dict,
        origin: List[int],
        cube_file: str,
        export_folder: str
) -> None:
    with open(export_folder + "/" + EXPORT_FILE, 'w') as file:

        file.write(f'load "{cube_file}"\n')
        file.write('background [255, 255, 255]\n')
        file.write('set perspectiveDepth OFF\n')
        file.write('isoSurface sign red blue cutoff 0.02""\n')
        file.write('color isosurface translucent\n')
        file.write('show isoSurface\n')
        file.write("bondRadiusMilliAngstroms = 40\n")
        file.write("select all and not _C and not _Ru; cpk 0.1\n")

        # draw_curve(export_data["curve"], file)
        draw_iso(export_data['pos_iso'], origin, file, 'blue', 1)
        draw_iso(export_data['neg_iso'], origin, file, 'red', 2)

        draw_plane(export_data['plane_data'], origin, file)

    print('done')


def main():
    pass


if __name__ == "__main__":
    exit(main())
