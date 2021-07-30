# Find Nodal Plane
Find the nodal plane of a helical molecular orbital.

utilities.py contains a function used to read .cube files that is modified from the [Atomic Simulation Codebase (ASE)](https://wiki.fysik.dtu.dk/ase/) codebase.

The "export" folder contains the source code for the helper file libread_cube.so. It's written in [Rust](https://www.rust-lang.org/).

Feel free to open an issue if you have any questions regarding the code or its usage!


## Example
If we want to analyze the highest occupied molecular orbital (HOMO) of an \[8\]cumulene, we can use the following command to find the helical nodal plane.

First, we have to make sure that the atom and the MO has been aligned along the z-axis.

Next, we need to setup up the following folder structure:
```bash
molecule_folder
└── data
    └── homo.cube
```
Running the script will then create a folder called `molecule_folder_homo` with all the analyzed data in it.

Next, we run the following command.
```bash
python find_nodal.py 8_cumulene/homo.cube 6 12 2 --carbon-chain 12,11,6,4,2
```
where "6" is the center atom, "12" is the bottom atom and "2" is the top atom. The atoms listed after the `--carbon-chain` command is the chain that the slices will be aligned against. This is important for molecules where the helical MO is not following a straight carbon chain (such as dimethyl-spiro[4.4]nonatetraene).

### Description of output files
All files that ends with `.npy` can be loaded with `np.load()`.
- `angles.npy`: Contains the angle between each succesive slice in units of degrees.
- `angles.png`: Plot of the cumulated sum of angles generated from the data in `angles.npy`
- `atom_info.npy`: Contains an array of `[atoms, top_carbon, bottom_carbon, [center_x, center_y, center_z]]`
    - `atoms`: An `Atoms` object containing information about the molecular system
    - `{top,bottom}_carbon`: Index of the top and bottom carbon, respectively.
    - `[center_x, center_y, center_z]`: A list of the x, y, and z-coordinate of the center atom.
- `fitted_{x,y}.npy`: Data points for the line that was fitted through the nodal plane. In other words, this data is the estimation of the nodal plane in a given slice.
- `homo.cube`: .cube file of the analyzed MO.
- `jmol_export.spt`: Contains information for [Jmol](http://wiki.jmol.org/index.php/Main_Page) to plot a visualization of the molecule, the analyzed MO, the found nodal plane as a yellow plane and the highest and lowest isovalue as a blue and red line, respectively.
- `jmol_plane_data.npy`: Information to plot the nodal plane in Jmol.
- `max_iso.npy`: The maximum isovalue in a given radius-filtered slice.
- `max_{neg,pos}_iso.npy`: Highest negative and positive isovalue in each radius-filtered slice. Is used to generate the Jmol visualization.
- `min_iso_val.npy`: The minimum isovalue in each radius-filtered slice.
- `phis.npy`: The angle between (0,0) and the maximum positive isovalue in a given radius-filtered slice.
- `planes.npy`: A list of planes sorted according to z-coordinate. Each plane contains a list of coordinates which contains x-, y-, z-coordinates and isovalue.
- `{x,y}_cross.npy`: The x-, and y-coordinates of where each slice goes from positive to negative. Calculated as the midway point between the closest positive and negative number.
- `xyz_vec.npy`: Contains the unit vectors of each dimension.

### Options
```bash
--show-slices
```
Show each slice that has been generated from the .cube file. Default is False. It will open each slice in a new matplotlib instance so be aware that it will not work if you're connected to a remote server. You also have to be aware that if the analyzed MO have been dumped on a fine grid, you'll open a lot of matplotlib-instances which might tank your computer.

It opens all plots at the same time so it will probably tank your computer if the MO have been dumped on a fine grid.

```bash
--export-gif
```
Export the slices as a .gif. Default is False. This option will try to export all slices used in the analysis as a .gif.

### Hückel example
The script`huckel_model.py` calculates the helicality of a chosen MO for a Hückel model of an [8]cumulene. The length of the cumulene can be extended by changing `dim`.
