#!/bin/bash
# Install OpenBLAS and MAGMA.

# verify installation of build tools
BINS="gcc gfortran nvcc"

for BIN in $BINS; do
	if [ ! $(which $BIN) ]; then
		>&2 echo "error: could not find $BIN"
		exit 1
	fi
done

# initialize directories, build parameters
INSTALL_PREFIX=$HOME/software
NUM_JOBS=$(nproc)

export CUDADIR=/usr/local/cuda
export MAGMADIR=$INSTALL_PREFIX/magma-2.2.0
export OPENBLASDIR=$INSTALL_PREFIX/OpenBLAS-0.2.19

mkdir -p $MAGMADIR $OPENBLASDIR

echo "Using $NUM_JOBS jobs"

# install OpenBLAS
echo "Building OpenBLAS..."

wget -q https://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz
tar -xf v0.2.19.tar.gz

cd OpenBLAS-0.2.19
cp lapack-netlib/make.inc.example lapack-netlib/make.inc
make -s -j $NUM_JOBS NO_LAPACK=0 TARGET=SANDYBRIDGE
make -s install PREFIX=$OPENBLASDIR

cd ..
rm -rf v0.2.19.tar.gz OpenBLAS-0.2.19

# install MAGMA
echo "Building MAGMA..."

wget -q http://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-2.2.0.tar.gz
tar -xf magma-2.2.0.tar.gz

cd magma-2.2.0
cp make.inc-examples/make.inc.openblas make.inc
make -s -j $NUM_JOBS
make -s install prefix=$MAGMADIR

cd ..
rm -rf magma-2.2.0.tar.gz magma-2.2.0

# complete installation
echo "Installation complete."
