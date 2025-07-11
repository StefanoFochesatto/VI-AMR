# VI-AMR/examples/glacier/

This directory contains a shallow ice approximation model of glaciers on land, which exploits and illustrates the tag-and-refine UDO and VCD VIAMR methods, along with gradient refinement.  See `METHOD.md` for the mathematical content.

## synthetic steady-state glaciers

### illustrations

The main code is `steady.py`, with formulas in `synthetic.py` and command-line argument processing in `clargs.py`.  To run a default steady-state glacier simulation for a synthetic "dome" glacier, with known exact solution, remember to activate the Firedrake virtual environment and then do
```
python3 steady.py
```
To get help add `-h`.  Here is an interesting bumpy bed and elevation-dependent surface mass balance case with Paraview-readable output, run in parallel:
```
mpiexec -n 4 python3 steady.py -prob cap -elevdepend -sELA 800 -m 20 -refine 4 -uniform 1 -opvd result_cap.pvd
```

### convergence and AMR efficiency

In the paper there is an example showing norm and geometrical errors against mesh complexity (number of elements).  Uniform, UDO+GR, and VCD+GR mesh refinement methods are shown.  Here are these runs:
```
mpiexec -n 12 python3 steady.py -newton -m 5 -refine 8 -uniform 8 -csv uniform.csv
mpiexec -n 12 python3 steady.py -newton -m 5 -refine 13 -csv udo.csv
mpiexec -n 12 python3 steady.py -newton -m 5 -refine 13 -vcd -csv vcd.csv
```

### bumpy bed examples

```
python3 steady.py -uniform 2 -refine 6 -prob cap -opvd result_cap.pvd
python3 steady.py -uniform 2 -refine 6 -prob range -opvd result_range.pvd
```

### elevation-dependent surface mass balance examples

```
mpiexec -n 4 python3 steady.py -prob cap -elevdepend -sELA 1000.0 -m 20 -uniform 1 -udo_n 2 -pcount 20 -refine 6 -opvd result_cap_1000.pvd
```
Vary `-sELA`, say 1000.0 -> 800.0 -> 600.0, to see increase in glaciation (inactive set area).


## realistic example which uses data for bed topography

FIXME: WIP

The NetCDF file `eastgr.nc` is already present.  It contains the ice-free bed topography for a portion of eastern Greenland, on a relatively low-resolution 5 km quadrilateral mesh.

To [read NetCDF files](https://unidata.github.io/netcdf4-python/) do
```
pip install netCDF4
```
Then do
```
python3 steady.py -data eastgr.nc -opvd result_data.pvd
```
Perhaps add options `-uniform 2 -refine 6 -pcount 20` etc.

### re-generating the NetCDF file

The file `eastgr.nc` _can be regenerated by_ cloning [PISM](https://github.com/pism/pism/) and using [NCO](https://nco.sourceforge.net/):
```
git clone https://github.com/pism/pism.git
sudo apt install nco
```
Now do the following:
```
cd pism/examples/std-greenland/
bash preprocess.sh   # downloads data; requires wget and the NCO
                     # generates pism_Greenland_5km_v1.1.nc
ncks -O -d x1,0.0,700.0e3 -d y1,-2.7e6,-1.3e6 -v topg pism_Greenland_5km_v1.1.nc eastgr.nc
```
One can view NetCDF files directly with [`ncview`](https://cirrus.ucsd.edu/ncview/).  I use the Debian packages for NCO and `ncview`.
