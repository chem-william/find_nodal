import numpy.linalg as la
from typing import List
from typing import Dict

EXPORT_FILE = "./jmol_export.spt"


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


def draw_plane(values: List[List[int]], origin: List[int], file) -> None:
    prev_coord = None
    line_str: str = ''
    for idx, value in enumerate(values):
        norm_xy = [value[0], value[1]]
        norm_xy = norm_xy/la.norm(norm_xy)

        coord = [origin[0] + norm_xy[0],
                 origin[1] + norm_xy[1],
                 origin[2] + value[2]]

        if idx > 1:
            line_str = f'draw plane{idx} PLANE {{{coord[0]} {coord[1]} {coord[2]}}} '  # Upper left
            line_str += f'{{{coord[0]} {coord[1]} {prev_coord[2]}}} '  # Upper right

            # line_str += f'{{{coord[0] - prev_xy[0]*2} {coord[1] - prev_xy[1]*2} {prev_coord[2]}}} '  # Lower right
            line_str += f'{{{coord[0]  - norm_xy[0]*2} {coord[1] - norm_xy[1]*2} {coord[2]}}} '  # Lower left

            file.write(line_str)
            file.write('\n')
            prev_coord = coord
        else:
            prev_coord = coord


def export_jmol(export_data: Dict, origin: List[int], cube_file: str) -> None:
    with open(EXPORT_FILE, 'w') as file:

        file.write(f'load "{cube_file}"\n')
        file.write('background [255, 255, 255]\n')
        file.write('set perspectiveDepth OFF\n')
        file.write('isoSurface sign red blue cutoff 0.02""\n')
        file.write('color isosurface translucent\n')
        file.write('show isoSurface\n')

        draw_iso(export_data['pos_iso'], origin, file, 'blue', 1)
        draw_iso(export_data['neg_iso'], origin, file, 'red', 2)

        draw_plane(export_data['plane_data'], origin, file)

        # for idx, line in enumerate(export_data):
        #     start = [origin[0], origin[1], line[2] + origin[2]]
        #     end = [line[0] + origin[0], line[1] + origin[1], line[2] + origin[2]]

        #     file.write((f"draw arr{idx} "
        #                 f"line width 0.2 "
        #                 f"{{{start[0]} {start[1]} {start[2]}}} "
        #                 f"{{{end[0]} {end[1]} {end[2]}}} "
        #                 f"color [255, 0, 0]\n"))
    print('done')


def main():
    pass


if __name__ == "__main__":
    exit(main())
