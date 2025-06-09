# examples/

## basic examples

The short programs `sphere.py`, `spiral.py`, and `aol.py` show many core abilities of the `VIAMR` class.  Each of these writes `.pvd` files for viewing in Paraview.

  * `sphere.py` refines an initially homogeneous mesh on a classical obstacle problem from Chapter 12 of Bueler (2021).  The three algoritms in the paper, namely UDO, VCD, and AVM, are used to mark elements for refinement near the free boundary, and refinement in the inactive set also occurs.  (Note that the AVM method depends on the [animate](https://github.com/mesh-adaptation/animate) library; see below.  To turn this off set `includeAVM` to `False` at the start of `sphere.py`.)  The default settings at the start of `sphere.py` are intended to generate a (more or less) apples-to-apples comparison of the methods.  View the `gap` variable in the output Paraview files to see the active, inactive, and free boundary sets.  See the `error` variable to see the distribution of numerical error.

  * `spiral.py` does a similar comparison on a classical obstacle problem from Graeser & Kornhuber (2009).  Only the UDO and VCD methods are demonstrated.

  * `aol.py` is a simple example that only does one level of refinement on a problem from M. Ainsworth, J.T. Oden, and C. Lee, Local a posteriori error estimators for variational inequalities, Numerical Methods for Partial Differential Equations 9 (1993), pp. 23–33.
  It is quoted in full in the paper, and used to produce a figure there.

## other examples

Programs `blister.py`, `parabola1d.py1`, `pollutant.py`, and `suttmeier.py` are documented only by their source code.

The `glaciers/` directory contains another example; see `glaciers/README.md` and `glaciers/METHOD.md` for what it is doing and how to run it.

### cleaning up

Clean up all `result*` files and subdirectories etc. with

```
make clean
```
