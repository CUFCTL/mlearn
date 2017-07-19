# libmlearn

Machine learning library written in C++ with GPU acceleration.

## Installation

Clone this repository and install dependencies by running `install-deps.sh`.

## Usage

To build library:
```
make install
```

To build and run tests:
```
cd test
make

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib test_image
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib test_matrix
```
