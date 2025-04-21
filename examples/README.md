# examples/

## basic examples

The short programs `spiral.py`, `sphere.py`, and `metric.py` show the core abilities of the `VIAMR` class.  Each of these writes a `.pvd` file for viewing in Paraview.

  * `spiral.py` does 4 refinements of an initially homogeneous mesh, on the classical obstacle problems defined by `SpiralObstacleProblem` from `viamr/utility.py`.  The `udomark()` method of `VIAMR` is used to mark elements for refinement near the free boundary.  View the `gap` variable in the output file to see the active, inactive, and free boundary sets.  To see the resulting mesh, and well-resolved spiral inactive set, view `gap` in the output file `result_spiral.pvd` in Paraview, with mesh edges shown.

  * `sphere.py` does a similar thing on `SphereObstacleProblem`, but it demonstrates the `vcesmark()` method, with 5 levels of refinement.  This method works in parallel.  In this problem the exact solution is known, so also see the `error` variable in the output file.

  * `metric.py` does a single refinement on `SphereObstacleProblem`.  This is done using the `metricrefine()` method of `VIAMR`, which depends on the [animate](https://github.com/mesh-adaptation/animate) library.  (See below.)  Again see the `gap` and `error` variables, but also compare the `error` variable with that from `sphere.py`.  The smaller maximum error here, from a mesh with lower complexity, emphasizes the role of refinement, here metric-based, in the inactive set.

## other examples

Programs `pollutant.py`, `BRinactive.py`, and `DWRinactive.py` are documented only by their source code.

## animate dependency

Basic example `metric.py` uses [animate](https://github.com/mesh-adaptation/animate) to do metric-based AMR.  For this, Ed had to do the following:

    git clone https://github.com/mesh-adaptation/animate.git
    cd animate/
    cp Makefile Makefile2
    curl -O https://raw.githubusercontent.com/mesh-adaptation/mesh-adaptation-docs/main/install/Makefile
    curl -O https://raw.githubusercontent.com/mesh-adaptation/mesh-adaptation-docs/main/install/petsc_configure_options.txt
    # hand edit Makefile: add this as line 33:
    #      FIREDRAKE_CONFIGURE_OPTIONS += "--netgen"
    unset PETSC_DIR; unset PYTHONPATH
    make install
    source firedrake-jan25/bin/activate   # or similar; activate the new (animate) venv
    make -f Makefile2 install             # uses "pip install -e ."
    pip install shapely siphash24
    cd ~/VI-AMR/                          # or similar
    pip install -e .

## glaciers/

See `glaciers/README.md` for how to run this example.

### cleaning up

Clean up all `result*` files and subdirectories with

```
make clean
```