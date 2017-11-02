# libmlearn

Machine learning library written in C++ with GPU acceleration.

## Features

Data types
- Image and Genome types included
- Plugin interface for creating your own data types

Feature extraction
- Principal Component Analysis
- Linear Discriminant Analysis
- Independent Component Analysis

Classification
- k-Nearest Neighbors
- Naive Bayes

Clustering
- k-means
- Gaussian mixture models

## Installation

Before running any commands, append these lines to `~/.bashrc`:
```
# CUDADIR should point to your CUDA installation
export CUDADIR=/usr/local/cuda
export PATH=$PATH:$CUDADIR/bin

export INSTALL_PREFIX=$HOME/software
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$INSTALL_PREFIX/include
export LIBRARY_PATH=$LIBRARY_PATH:$INSTALL_PREFIX/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INSTALL_PREFIX/lib
```

You should then be able to install libmlearn and its dependencies:
```
# clone repository
git clone https://github.com/CUFCTL/libmlearn.git
cd libmlearn

# install OpenBLAS and MAGMA
make install-deps -j [num-jobs]

# install library
make -j [num-jobs]
```

## Usage

Refer to the test programs in the `test` folder for example uses of libmlearn:
```
make examples

cd test

build/test-classification
build/test-clustering
build/test-data
build/test-matrix
```
