# VI-AMR

This repository contains algorithms for adaptive mesh refinement with a goal of targeting refinement to the free boundary. These codes support the research presented in my thesis.

There are two methods, called Unstructured Dilation Operator (UDO) and Varable Coefficient Elliptic Smoothing (VCES). These are element marking methods of the class `VIAMR`, which is implemented in `viamr/viamr.py`.

## Dependencies

To get started, install firedrake along with the netgen/ngsolve integration. Follow the instructions at [Firedrake download page](https://www.firedrakeproject.org/firedrake/download.html).

On an new firedrake install, netgen/ngsolve integration is added by running the firedrake-install script with the `--netgen` flag.

```
python3 firedrake-install --netgen
```

An existing firedrake install can be updated to include netgen/ngsolve integration by

```
(firedrake) python3 firedrake-update --netgen
```

from running the firedrake-update script in `firedrake/bin/`. Make sure the firedrake virtual environment is active before doing so.

## Installation

### development

Install editable with pip:

```
pip install -e .
```

### production

Install with pip:

```
pip install .
```

## Usage

A first example compares 3 levels of refinement with UDO and VCES, using default parameters, on two obstacle problems. The sphere problem has a known exact solution while the spiral problem does not. In `examples/` do

```
python3 spherespiral.py
```

Then view the fields in `result_PROBLEM_METHOD.pvd` using [Paraview](FIXME LINK). These files contain the obstacle `psi`, the solution `u`, and the gap `u - psi`. The `_sphere` files also contain the numerical error `|u - uexact|`.

Clean up all `results...` files and subdirectories with

```
make clean
```

## Testing

Software tests use [pytest](https://docs.pytest.org/en/stable/index.html). In the main directory `VI-AMR/` do

```
pytest .
```

FIXME from here
