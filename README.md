# libmlearn

Machine learning library written in C++ with GPU acceleration.

## Installation

Clone this repository and install dependencies by running `install-deps.sh`.

## Usage

To build library and tests:
```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=. -DGPU=1
make install
```

To run tests:
```
cd build/test

./test_image [infile] [outfile]
./test_matrix
```
