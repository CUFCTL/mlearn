# define build parameters
DEBUG ?= 0
GPU   ?= 0
INSTALL_PREFIX ?= .

# define compiler suite
CXX  = g++

ifeq ($(GPU), 1)
NVCC = nvcc
else
NVCC = g++
endif

# define include paths
CUDADIR     ?= /usr/local/cuda
MAGMADIR    ?= $(HOME)/software/magma-2.2.0
OPENBLASDIR ?= $(HOME)/software/OpenBLAS-0.2.19
SRCDIR      = src

# define compiler flags, libraries
LDFLAGS   = -shared -lm
CXXFLAGS  = -std=c++11 -fPIC \
            -I$(OPENBLASDIR)/include \
            -I$(SRCDIR)
NVCCFLAGS = -std=c++11 \
            -I$(CUDADIR)/include \
            -I$(MAGMADIR)/include \
            -I$(OPENBLASDIR)/include \
            -I$(SRCDIR) \
            -Wno-deprecated-gpu-targets

ifeq ($(DEBUG), 1)
CXXFLAGS  += -g -pg -Wall
NVCCFLAGS += -g -pg -Xcompiler -Wall
else
CXXFLAGS  += -O3 -Wno-unused-result
NVCCFLAGS += -O3 -Xcompiler -Wno-unused-result
endif

ifeq ($(GPU), 1)
LDFLAGS   += -L$(MAGMADIR)/lib -lmagma \
             -L$(CUDADIR)/lib64 -lcudart -lcublas \
             -L$(OPENBLASDIR)/lib -lopenblas
else
LDFLAGS   += -L$(OPENBLASDIR)/lib -lopenblas
NVCCFLAGS = $(CXXFLAGS)
endif

# define binary targets
OBJS = $(addprefix $(SRCDIR)/, \
	classifier/bayes.o \
	classifier/knn.o \
	data/dataset.o \
	data/image.o \
	feature/ica.o \
	feature/identity.o \
	feature/lda.o \
	feature/pca.o \
	math/math_utils.o \
	math/matrix.o \
	math/matrix_utils.o \
	model/model.o \
	util/logger.o \
	util/timer.o )

BINS = libmlearn.so

all: echo $(BINS)

echo:
	$(info DEBUG     = $(DEBUG))
	$(info GPU       = $(GPU))
	$(info CXX       = $(CXX))
	$(info NVCC      = $(NVCC))
	$(info LDFLAGS   = $(LDFLAGS))
	$(info CXXFLAGS  = $(CXXFLAGS))
	$(info NVCCFLAGS = $(NVCCFLAGS))

%.o: %.cpp %.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

libmlearn.so: $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

install: all
	mkdir -p $(INSTALL_PREFIX)/include/classifier
	mkdir -p $(INSTALL_PREFIX)/include/data
	mkdir -p $(INSTALL_PREFIX)/include/feature
	mkdir -p $(INSTALL_PREFIX)/include/math
	mkdir -p $(INSTALL_PREFIX)/include/model
	mkdir -p $(INSTALL_PREFIX)/include/util
	mkdir -p $(INSTALL_PREFIX)/lib
	cp $(SRCDIR)/classifier/*.h $(INSTALL_PREFIX)/include/classifier
	cp $(SRCDIR)/data/*.h $(INSTALL_PREFIX)/include/data
	cp $(SRCDIR)/feature/*.h $(INSTALL_PREFIX)/include/feature
	cp $(SRCDIR)/math/*.h $(INSTALL_PREFIX)/include/math
	cp $(SRCDIR)/model/*.h $(INSTALL_PREFIX)/include/model
	cp $(SRCDIR)/util/*.h $(INSTALL_PREFIX)/include/util
	cp $(SRCDIR)/mlearn.h $(INSTALL_PREFIX)/include
	cp $(BINS) $(INSTALL_PREFIX)/lib

clean:
	rm -rf $(OBJS) $(BINS) gmon.out
