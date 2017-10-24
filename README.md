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

Before running any commands, you should add the path to your CUDA installation to `~/.bashrc`:
```
# Ubuntu
export CUDADIR=/usr/local/cuda
export PATH=$PATH:$CUDADIR/bin

# Palmetto (example)
export CUDADIR=/software/cuda-toolkit/7.5.18
export PATH=$PATH:$CUDADIR/bin
```

You should then be able to install mlearn and its dependencies:
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

To run tests:
```
bin/test-classification
bin/test-clustering [k ...]
bin/test-data [type] [infile] [outfile]
bin/test-matrix
```

To include this library in your project, you need to add the library folder to `LD_LIBRARY_PATH`:
```
export MLEARNDIR=$HOME/software/libmlearn
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MLEARNDIR/lib

export OPENBLASDIR=$HOME/software/OpenBLAS-0.2.19
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENBLASDIR/lib
```

Append the above lines to `~/.bashrc` for ease of use.
