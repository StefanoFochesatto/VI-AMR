# VI-AMR

This repository contains algorithms for adaptive mesh refinement for variational inequalities, that is, for free boundary problems. The goal is to have targeted refinement near a computed free boundary.

There are two element marking methods, Unstructured Dilation Operator (UDO) and Varable Coefficient Elliptic Smoothing (VCES). These are methods of the class `VIAMR`, which is implemented in `viamr/viamr.py`.

These codes support S. Fochesatto (2024). _Adaptive mesh refinement for variational inequalities_, Master of Science project, UAF, and a paper in progress.

## Dependencies

To get started, install firedrake with the [netgen/ngsolve](https://ngsolve.org/) integration. To do this install, follow the instructions at [Firedrake download page](https://www.firedrakeproject.org/firedrake/download.html). Run the `firedrake-install` script with the `--netgen` flag:

```
python3 firedrake-install --netgen
```

For an existing firedrake install, make sure that the firedrake virtual environment is active, and then use the `firedrake-update` script from `firedrake/bin/` to add netgen/ngsolve integration:

```
python3 firedrake-update --netgen
```

Now activate the venv. Typically something like:

```
unset PETSC_DIR;  unset PYTHONPATH
source ~/firedrake/bin/activate
```

Now install [shapely](https://pypi.org/project/shapely/) in the venv:

```
pip install shapely
pip install siphash24   # may also be needed
```

## Installation of VI-AMR

### development install

Install editable with pip:

```
pip install -e .
```

### production install

Install with pip:

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

A first example compares 3 levels of refinement with UDO and VCES, using default parameters, on two obstacle problems. The sphere problem has a known exact solution while the spiral problem does not. First make sure that the firedrake virtual environment is active. Then do

```
cd examples/
python3 spherespiral.py
```

View the output fields in `result_PROBLEM_METHOD.pvd` using [Paraview](https://www.paraview.org/). These files contain the obstacle `psi`, the solution `u`, and the gap `u - psi`. The `_sphere` files also contain the numerical error `|u - uexact|`.

Clean up all `results*` files and subdirectories with

```
make clean
```

## Testing

Software tests use [pytest](https://docs.pytest.org/en/stable/index.html). In the main directory `VI-AMR/` do

```
pytest .
```
