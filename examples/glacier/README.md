# VI-AMR/examples/glacier/

## synthetic examples

Remember to activate the Firedrake virtual environment.

To run a default steady-state glacier simulation for a synthetic glacier do
```
python3 steady.py
```
or for a more interesting case with Paraview-readable output,
```
python3 steady.py -refine 4 -prob range -opvd result_range.pvd
```

## an example which uses data for bed topography

The NetCDF file `eastgr.nc` is already present.  It contains the ice-free bed topography for a portion of eastern Greenland, on a relatively low-resolution 5 km quadrilateral mesh.

To [read NetCDF files](https://unidata.github.io/netcdf4-python/) do
```
pip install netCDF4
```
Then do
```
python3 steady.py -data eastgr.nc
```

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
