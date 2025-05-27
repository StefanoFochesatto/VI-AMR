# examples/

## basic examples

The short programs `sphere.py` and `spiral.py` show many core abilities of the `VIAMR` class.  Each of these writes `.pvd` files for viewing in Paraview.

  * `sphere.py` refines an initially homogeneous mesh on a classical obstacle problem from Chapter 12 of Bueler (2021).  The three algoritms in the paper, namely UDO, VCD, and AVM, are used to mark elements for refinement near the free boundary, and refinement in the inactive set also occurs.  (Note that the AVM method depends on the [animate](https://github.com/mesh-adaptation/animate) library; see below.  To turn this off set `includeAVM` to `False` at the start of `sphere.py`.)  The default settings at the start of `sphere.py` are intended to generate a (more or less) apples-to-apples comparison of the methods.  View the `gap` variable in the output Paraview files to see the active, inactive, and free boundary sets.  See the `error` variable to see the distribution of numerical error.

  * `spiral.py` does a similar comparison on a classical obstacle problem from Graeser & Kornhuber (2009).  Only the UDO and VCD methods are demonstrated.

## other examples

Programs `blister.py`, `pollutant.py`, and `suttmeier.py` are documented only by their source code.  Also, the `glaciers/` directory contains another example; see `glaciers/README.md` for what it is doing and how to run it.

## animate dependency

FIXME THIS MAY BE COMPLETELY WRONG AND OUT OF DATE.  FOLLOW THE INSTALL INSTRUCTIONS AT THE animate PAGE?

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

### cleaning up

Clean up all `result*` files and subdirectories with

```
make clean
```