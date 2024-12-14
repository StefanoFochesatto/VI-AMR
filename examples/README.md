# VI-AMR/examples/

For `spherespiral.py` and `glacier.py` there are no additional dependencies.

However, `metricaveraging.py` uses [animate](https://github.com/mesh-adaptation/animate) to do metric-based AMR.  For this, Ed had to do the following:

    git clone https://github.com/mesh-adaptation/animate.git
    cd animate/
    cp Makefile Makefile2
    curl -O https://raw.githubusercontent.com/mesh-adaptation/mesh-adaptation-docs/main/install/Makefile
    curl -O https://raw.githubusercontent.com/mesh-adaptation/mesh-adaptation-docs/main/install/petsc_configure_options.txt
    # hand edit Makefile: add this as line 33:
    #      FIREDRAKE_CONFIGURE_OPTIONS += "--netgen"
    unset PETSC_DIR; unset PYTHONPATH
    make install
    source bin/activate           # or similar; activate the new (animate) venv
    make -f Makefile2 install     # uses "pip install -e ."
    pip install shapely
    pip install siphash24
    cd ~/VI-AMR/                  # or similar
    pip install -e .
