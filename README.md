# Find Nodal Plane
Find the nodal plane of a helical molecular orbital.

utilities.py contains a function used to read .cube files that is modified from the [Atomic Simulation Codebase (ASE)](https://wiki.fysik.dtu.dk/ase/) codebase.

The "export" folder contains the source code for the helper file libread_cube.so. It's written in [Rust](https://www.rust-lang.org/).

Feel free to open an issue if you have any questions regarding the code or its usage!


## Example
If we want to analyze the highest occupied molecular orbital (HOMO) of an \[8\]cumulene, we can use the following command to find the helical nodal plane.

First, we have to make sure that the atom and the MO has been aligned along the z-axis.
Next, we run the following command.
```bash
python find_nodal.py 8_cumulene/homo.cube 6 12 2 --carbon-chain 12,11,6,4,2
```
where "6" is the center atom, "12" is the bottom atom and "2" is the top atom. The atoms listed after the `--carbon-chain` command is the chain that the slices will be aligned against. This is important for molecules where the helical MO is not following a straight carbon chain (such as dimethyl-spiro[4.4]nonatetraene).

### Options
```bash
--show-slices
```
Show each slice that has been generated from the .cube file. Default is False. It will open each slice in a new matplotlib instance so be aware that it will not work on a cluster.

It opens all plots at the same time so it will probably tank your computer if the MO have been dumped on a fine grid.

```bash
--export-gif
```
Export the slices as a .gif. Default is False. This option will try to export all slices used in the analysis as a .gif.
