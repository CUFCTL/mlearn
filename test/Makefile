# define build parameters
DEBUG ?= 0

# define compiler suite
CXX = g++

# define include paths
MLEARNDIR ?= ..

# define compiler flags, libraries
LDFLAGS   = -lm -L$(MLEARNDIR)/lib -lmlearn
CXXFLAGS  = -std=c++11 -I$(MLEARNDIR)/include

ifeq ($(DEBUG), 1)
CXXFLAGS  += -g -pg -Wall
else
CXXFLAGS  += -O3 -Wno-unused-result
endif

# define binary targets
BINS = \
	test_image \
	test_matrix

all: $(BINS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -rf *.o $(BINS) gmon.out
