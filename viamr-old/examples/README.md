# examples/

## basic examples

The short programs `aol.py`, `sphere.py`, `spiral.py`, `blisters.py`, and `lshaped.py`, which all solve classical Poisson/Laplacian obstacle problems, show many core abilities of the `VIAMR` class.  Each of these writes `.pvd` files for viewing in Paraview.

  * `aol.py` is a simple example that only does one level of VCD free-boundary refinement on a problem from M. Ainsworth, J.T. Oden, and C. Lee, _Local a posteriori error estimators for variational inequalities_, Numerical Methods for Partial Differential Equations 9 (1993), pp. 23–33.  It is quoted in full in the paper, and used to produce a figure there.  It does not refine in the inactive set, which is needed for convergence.

  * `sphere.py` refines an initially homogeneous mesh on a problem from Chapter 12 of Bueler (2021).  The three algorithms in the paper, UDO+BR, VCD+BR, and AVM, are used to mark elements for refinement near the free boundary and in the inactive set.  (Note that the AVM method depends on the [animate](https://github.com/mesh-adaptation/animate) library; see below.)  The default settings at the start of `sphere.py` are intended to generate a (more or less) apples-to-apples comparison of the methods, which are shown as results in the paper.  View the `gap` variable in the output Paraview files to see the active, inactive, and free boundary sets.  See the `error` variable to see the distribution of numerical error.

  * `spiral.py` does a comparison on a classical obstacle problem from Graeser & Kornhuber (2009).  Here the active set is small but the free boundary is big (long).  Only the UDO+BR and VCD+BR methods are demonstrated.

  * `blisters.py` uses VCD+BR on an example with a large active set, and with the blistering property.  The AMR technique is very efficient relative to uniform refinement.

  * `lshaped.py` uses the AVM on an example with both a free boundary and a classic interior corner.  The AMR technique addresses both.  This example was written primarily with Claude AI (Summer 2025).  Users can generate VIAMR examples by vibe coding if they so desire.

## other examples

Programs `parabola1d.py1`, `pollutant.py`, and `suttmeier.py` are documented only by their source code.

The `glaciers/` directory contains another example; see `glaciers/README.md` and `glaciers/METHOD.md` for what it is doing and how to run it.

### cleaning up

Clean up all `result*` files and subdirectories etc. with

```
make clean
```
