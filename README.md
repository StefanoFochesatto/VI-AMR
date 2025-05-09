# VI-AMR

This repository contains algorithms for adaptive mesh refinement for variational inequalities (VIs).  The methods require the constraint set to be defined by a lower-bound inequality, that is, they are for unilateral obstacle problems.  A primary goal is to have targeted refinement near a computed free boundary, and to be able to measure location errors in the free boundary.  Refinement in the inactive set using PDE-type error estimators is also supported.

We define the class `VIAMR` in `viamr/viamr.py`.  It provides two element-marking methods, Unstructured Dilation Operator (UDO) and Varable Coefficient Diffusion (VCD), for use with tag-and-refine adaptive mesh refinement (AMR).  The class also provides a method to refine in the inactive set using the Babuška-Rheinboldt (BR) residual error indicator, a method for anisotropic metric-based refinement which also prefers refinement near the free boundary, and geometric methods for measuring active set convergence.

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

## Installation of VI-AMR

### development install

From the VI-AMR directory, install editable with pip:

```
pip install -e .
```

### production install

From the VI-AMR directory, install with pip:

```
pip install .
```

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

## Testing

Software tests use [pytest](https://docs.pytest.org/en/stable/index.html). In the main directory `VI-AMR/` do

```
pytest .
```
