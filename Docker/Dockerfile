# Dockerfile for installing Firedrake with ALL mesh adaptation support via Mmg, ParMmg, Netgen, Animate, Movement, and Goalie
# Modified from: https://github.com/mesh-adaptation/docs/blob/main/docker/Dockerfile.firedrake-parmmg
# Thanks Davor!

# Dockerfile for installing Firedrake with mesh adaptation support via Mmg and ParMmg,
# as well as Animate, Goalie, and Movement

# First stage: Build MPICH and PETSc. Based off
# https://github.com/firedrakeproject/firedrake/blob/master/docker/Dockerfile.env
FROM ubuntu:24.04 AS petscbuild

USER root
WORKDIR /home/firedrake
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
        curl vim docker.io openssh-client build-essential autoconf automake \
        cmake gfortran git libopenblas-serial-dev libtool python3-dev python3-pip python3-tk \
        python3-venv python3-requests zlib1g-dev libboost-dev sudo gmsh bison flex ninja-build \
        libocct-ocaf-dev libocct-data-exchange-dev swig graphviz libcurl4-openssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Fetch PETSc
RUN git clone --depth 1 https://github.com/firedrakeproject/petsc.git

# Build MPICH manually because we don't want PETSc to build it twice
RUN bash -c 'export PACKAGES=/home/firedrake/petsc/packages; \
    cd petsc; \
    ./configure \
        --COPTFLAGS=-O3 -march=native -mtune=native \
        --CXXOPTFLAGS=-O3 -march=native -mtune=native \
        --FOPTFLAGS=-O3 -march=native -mtune=native \
        --with-c2html=0 \
        --with-debugging=0 \
        --with-fortran-bindings=0 \
        --with-make-np=12 \
        --with-shared-libraries=1 \
        --with-zlib \
        --download-eigen \
        --download-fftw \
        --download-hdf5 \
        --download-hwloc \
        --download-hypre \
        --download-metis \
        --download-mmg \
        --download-mumps \
        --download-mpich \
        --download-mpich-device=ch3:sock \
        --download-netcdf \
        --download-parmmg \
        --download-pastix \
        --download-pnetcdf \
        --download-ptscotch \
        --download-scalapack \
        --download-suitesparse \
        --download-superlu_dist \
        PETSC_ARCH=packages; \
        mv ${PACKAGES}/include/petscconf.h ${PACKAGES}/include/old_petscconf.nope;' && \
    rm -rf /home/firedrake/petsc/**/externalpackages && \
    rm -rf /home/firedrake/petsc/src/docs && \
    rm -f /home/firedrake/petsc/src/**/tutorials/output/* && \
    rm -f /home/firedrake/petsc/src/**/tests/output/*
# Don't run make here, we only want MPICH and HWLOC
# It is also necessary to move `petscconf.h` so packages isn't treated like a working PETSc

# Build PETSc
RUN bash -c 'export PACKAGES=/home/firedrake/petsc/packages; \
    cd petsc; \
    ./configure \
        --COPTFLAGS=-O3 -march=native -mtune=native \
        --CXXOPTFLAGS=-O3 -march=native -mtune=native \
        --FOPTFLAGS=-O3 -march=native -mtune=native \
        --with-c2html=0 \
        --with-debugging=0 \
        --with-fortran-bindings=0 \
        --with-make-np=12 \
        --with-shared-libraries=1 \
        --with-bison \
        --with-flex \
        --with-zlib \
        --with-eigen-dir=${PACKAGES} \
        --with-fftw-dir=${PACKAGES} \
        --with-hdf5-dir=${PACKAGES} \
        --with-hwloc-dir=${PACKAGES} \
        --with-hypre-dir=${PACKAGES} \
        --with-metis-dir=${PACKAGES} \
        --with-mmg-dir=${PACKAGES} \
        --with-mpi-dir=${PACKAGES} \
        --with-mumps-dir=${PACKAGES} \
        --with-netcdf-dir=${PACKAGES} \
        --with-parmmg-dir=${PACKAGES} \
        --with-pastix-dir=${PACKAGES} \
        --with-pnetcdf-dir=${PACKAGES} \
        --with-ptscotch-dir=${PACKAGES} \
        --with-scalapack-dir=${PACKAGES} \
        --with-suitesparse-dir=${PACKAGES} \
        --with-superlu_dist-dir=${PACKAGES} \
        PETSC_ARCH=default; \
    make PETSC_DIR=/home/firedrake/petsc PETSC_ARCH=default all; \
    mv ${PACKAGES}/include/eigen3/Eigen ${PACKAGES}/include' && \
    rm -rf /home/firedrake/petsc/**/externalpackages && \
    rm -rf /home/firedrake/petsc/src/docs && \
    rm -f /home/firedrake/petsc/src/**/tutorials/output/* && \
    rm -f /home/firedrake/petsc/src/**/tests/output/*

# Final stage: Install Firedrake and Animate, Movement, Goalie
FROM ubuntu:24.04

RUN apt-get update \
    && apt-get -y dist-upgrade \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
        curl vim docker.io openssh-client build-essential autoconf automake \
        cmake gfortran git libopenblas-serial-dev libtool python3-dev python3-pip python3-tk \
        python3-venv python3-requests zlib1g-dev libboost-dev sudo gmsh bison flex ninja-build \
        libocct-ocaf-dev libocct-data-exchange-dev swig graphviz libcurl4-openssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Change the `ubuntu` user to `firedrake`
# and ensure that we do not run as root on self-hosted systems
RUN usermod -d /home/firedrake -m ubuntu \
    && usermod -l firedrake ubuntu \
    && groupmod -n firedrake ubuntu \
    && usermod -aG sudo firedrake \
    && echo "firedrake:docker" | chpasswd \
    && echo "firedrake ALL=(ALL) NOPASSWD: ALL" >>/etc/sudoers \
    && ldconfig

USER firedrake
WORKDIR /home/firedrake

# Copy PETSc and MPI directories from petscbuild stage
COPY --from=petscbuild /home/firedrake/petsc /home/firedrake/petsc

# Set some useful environment variables
ENV LC_ALL=C.UTF-8
ENV PETSC_ARCH=default
ENV PETSC_DIR=/home/firedrake/petsc
ENV MPICH_DIR=${PETSC_DIR}/packages/bin
ENV HDF5_DIR=${PETSC_DIR}/packages
ENV HDF5_MPI=ON
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Install Firedrake and Animate, Movement, Goalie
RUN curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-install && \
    bash -c "python3 firedrake-install \
		--no-package-manager \
		--disable-ssh \
		--honour-petsc-dir \
		--mpihome=${MPICH_DIR} \
		--mpicc=${MPICH_DIR}/mpicc \
		--mpicxx=${MPICH_DIR}/mpicxx \
		--mpif90=${MPICH_DIR}/mpif90 \
		--mpiexec=${MPICH_DIR}/mpiexec \
        --netgen \
		--install thetis && \
    source firedrake/bin/activate && \
    cd firedrake/src && \
    git clone https://github.com/mesh-adaptation/animate.git && \
    git clone https://github.com/mesh-adaptation/movement.git && \
    git clone https://github.com/mesh-adaptation/goalie.git && \
    cd animate && make install && \
    cd ../movement && make install && \
    cd ../goalie && make install && \
    pip install shapely siphash24 pandas matplotlib && \
    cd ../ && git clone https://github.com/StefanoFochesatto/VI-AMR.git && \
    cd VI-AMR && \
    pip install . "


