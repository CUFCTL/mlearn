# mlearn

A condensed machine learning library written in C++/CUDA.

## Features

Data types
- Image and Genome types included
- Plugin interface for creating your own data types

Dimensionality reduction
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

This project depends on CUDA. The CUDA Toolkit can be downloaded [here](https://developer.nvidia.com/cuda-downloads).

Before running any commands, append these lines to `~/.bashrc`:
```
# CUDADIR should point to your CUDA installation
export CUDADIR="/usr/local/cuda"
export PATH="$CUDADIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDADIR/lib64:$LD_LIBRARY_PATH"

export INSTALL_PREFIX="$HOME/software"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"
```

You should then be able to install mlearn:
```
# clone repository
git clone https://github.com/CUFCTL/mlearn.git
cd mlearn

# install library
make -j [num-jobs]
```

## Usage

Refer to the test programs in the `test` folder for example uses of mlearn:
```
make examples

cd test

build/test-classification
build/test-clustering
build/test-data
build/test-matrix
```
