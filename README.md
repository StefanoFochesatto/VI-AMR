# VIAMR

This repository contains algorithms for adaptive mesh refinement (AMR) for variational inequalities (VIs).  The methods require the constraint set to be defined by a lower- and upper-bound inequalities.  Primary goals are to have targeted refinement near computed free boundaries, and to be able to measure location errors in free boundaries.  Refinement in the inactive set using a couple of classical PDE-type error estimators is supported.

We define the class `VIAMR` in `viamr/viamr.py`.  It provides two element-marking methods based on adjacency to the computed free boundary, namely Unstructured Dilation Operator (UDO) and Varable Coefficient Diffusion (VCD).  The class also provides two methods to refine in the inactive set, namely gradient recovery and the Babuška-Rheinboldt (BR) residual error indicator.  These are all for use with tag-and-refine mesh refinement.

A metric-based refinement method is also implemented, which combines anisotropic Hessian-based information with adjaceny to the free boundary.

These codes support S. Fochesatto (2024). _Adaptive mesh refinement for variational inequalities_, Master of Science project, UAF, and a paper in progress.

## Dependencies

To get started, install Firedrake following the instructions at the [Firedrake install page](https://www.firedrakeproject.org/install.html#).

Now activate the virtual environment (venv). Typically something like:

```
source ~/venv-firedrake/bin/activate
```

Now pip install [shapely](https://pypi.org/project/shapely/), siphash24, vtk, and [ngspetsc](https://github.com/NGSolve/ngsPETSc) in the venv:

```
pip install shapely siphash24 vtk ngspetsc
```

To use metric-based refinement, the [animate](https://github.com/mesh-adaptation/animate) adaptive mesh refinement library is used.  To install this do
```
git clone https://github.com/mesh-adaptation/animate.git
python3 -m pip install -e animate
```

## Installation

### clone the VIAMR repository and enter the directory

```
git clone https://github.com/StefanoFochesatto/VI-AMR.git
cd VI-AMR/
```

### install

Either install editable with pip:

```
pip install -e .
```
or plain:

```
pip install .
```

#### Build notes May 2025

When running a code which depends on animate, such as `examples/sphere.py`, you might get an error which ends with
```
...
[0] Grid adaptor mmg not registered; you may need to add --download-mmg to your ./configure options
```
This error can be resolved by adding `--download-mmg --download-parmmg` to your PETSc configuration, and recompiling PETSc.  For example, go to the `petsc/` directory in your Firedrake installation and do
```
python3 ../firedrake-configure --show-petsc-configure-options | xargs -L1 ./configure --download-metis --download-parmetis --download-mmg --download-parmmg --download-eigen
```
Then do `make all` and perhaps `make check`.

### Using Docker

A docker image is available with most of the setup compete. To get started ensure that you have [Docker](https://docs.docker.com/engine/install/) installed and running on your system.

Pull the Docker image:

```
docker pull stefanofochesatto/viamr:latest
```

Run a Docker container from the image:

```
docker run --rm -it -v ${HOME}:${HOME} stefanofochesatto/viamr:latest
```

FYI: The `--rm` flag will remove the container once it exits. The `-it` flag runs the container with an interactive shell environment (`ctrl + d` to exit). Finally `-v ${HOME}:${HOME}` is giving the container access to your `HOME` directory so you can navigate your files within the interactive shell environment.

Once the docker container is up and running, you can activate the python environment as usual. You'll also want to reinstall VI-AMR as the docker image was built with a previous version of the library. (automatic builds are low priority)

## Usage

These basic examples demonstrate refinement with VCD and UDO, respectively.  First make sure that the firedrake virtual environment is active.  Then do:

```
cd examples/
python3 sphere.py
python3 spiral.py
```

The sphere problem has a known exact solution while the spiral problem does not.

View the output fields in `result_*.pvd` using [Paraview](https://www.paraview.org/).  These files contain the obstacle `psi`, the solution `u`, and the gap `u - psi`. The `result_sphere.pvd` file also contains the numerical error `|u - uexact|`.

## Generating meshes

Meshes can be created using the [Firedrake utility mesh generators](https://www.firedrakeproject.org/_modules/firedrake/utility_meshes.html).  Alternatively, one can create Netgen meshes with e.g. `SplineGeometry().GenerateMesh()`.  The resulting meshes have different refinement capabilities.  Future bug fixes and feature improvements in Netgen, ngsPETSc, and PETSc DMPlex might change this situation, but for now see the known limitations below.

## Known limitations

  1. [PETSc's DMPlex mesh transformations](https://petsc.org/release/overview/plex_transform_table/) include skeleton based refinement (SBR) in 2D, but [currently SBR is not available in 3D](https://petsc.org/release/src/dm/impls/plex/transform/impls/refine/sbr/plexrefsbr.c.html).  This limits `VIAMR.refinemarkedelements()` to applications in 2D.
  1. Parallel application of `VIAMR.udomark()` requires that mesh distribution parameters be explicitly set.  For example, when using a utility mesh:
      ```
      UnitSquareMesh(m0, m0, distribution_parameters={"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)})
      ```
  1. `VIAMR.adaptaveragedmetric()` and `VIAMR.vcdmark()` are known to generate different results in serial and parallel.  See [issue #37](https://github.com/StefanoFochesatto/VI-AMR/issues/37) and [issue #38](https://github.com/StefanoFochesatto/VI-AMR/issues/38), respectively.
  1. For the reason given on [this issue](https://github.com/firedrakeproject/mpi-pytest/issues/13), use of [pytest](https://docs.pytest.org/en/stable/) cannot easily be extended to parallel using the [mpi-pytest](https://github.com/firedrakeproject/mpi-pytest) plugin.  Thus parallel regression testing is manual; see the bottom of this page.  Future bug fixes by the mpi-pytest developers could fix this.
  1. `VIAMR.jaccard()` only works in parallel if one mesh is a submesh of the other.  See the doc string.
  1. `VIAMR.hausdorff()` does not work in parallel.  Also, it is the only part of VIAMR which depends on the [shapely](https://pypi.org/project/shapely/) library.

## Testing

Serial software tests use [pytest](https://docs.pytest.org/en/stable/index.html). In the main directory `VI-AMR/` do
```
pytest .
```

For an HTML coverage report from these serial tests do:
```
pip install pytest-cov
pytest --cov-report html --cov=viamr tests/
firefox htmlcov/index.html
```

For parallel tests do the following or similar:
```
cd tests/
mpiexec -n 3 python3 test_parallel.py
```
Success is completion without any error messages.
