BUILD = build
BUILD_EX = test/build
SRC = src

DEBUG ?= 0
INSTALL_PREFIX ?= $(HOME)/software

CMAKEFLAGS = -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX)

ifeq ($(DEBUG), 1)
CMAKEFLAGS += -DCMAKE_BUILD_TYPE="Debug"
else
CMAKEFLAGS += -DCMAKE_BUILD_TYPE="Release"
endif

all: mlearn examples
.FORCE:

install-openblas:
	@echo -e "\nInstalling OpenBLAS...\n"

	mkdir -p $(INSTALL_PREFIX)
	wget -q https://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz
	tar -xf v0.2.19.tar.gz

	cd OpenBLAS-0.2.19 && cp lapack-netlib/make.inc.example lapack-netlib/make.inc
	+$(MAKE) -s -C OpenBLAS-0.2.19 NO_LAPACK=0 TARGET=NEHALEM
	+$(MAKE) -s -C OpenBLAS-0.2.19 install PREFIX=$(INSTALL_PREFIX)

	rm -rf v0.2.19.tar.gz OpenBLAS-0.2.19

install-magma:
	@echo -e "\nInstalling MAGMA...\n"

	mkdir -p $(INSTALL_PREFIX)
	wget -q http://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-2.2.0.tar.gz
	tar -xf magma-2.2.0.tar.gz

	cd magma-2.2.0 && cp make.inc-examples/make.inc.openblas make.inc
	+$(MAKE) -s -C magma-2.2.0 OPENBLASDIR=$(INSTALL_PREFIX)
	+$(MAKE) -s -C magma-2.2.0 install OPENBLASDIR=$(INSTALL_PREFIX) prefix=$(INSTALL_PREFIX)

	rm -rf magma-2.2.0.tar.gz magma-2.2.0

install-deps: install-openblas install-magma
	@echo -e "\nInstallation complete.\n"

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD_EX):
	mkdir -p $(BUILD_EX)

mlearn: $(SRC)/mlearn/**/*.h $(SRC)/mlearn/**/*.cpp | $(BUILD)
	cd $(BUILD) && cmake .. $(CMAKEFLAGS)
	+$(MAKE) -C $(BUILD) install

examples: mlearn | $(BUILD_EX)
	cd $(BUILD_EX) && cmake .. $(CMAKEFLAGS)
	+$(MAKE) -C $(BUILD_EX)

clean:
	rm -rf $(BUILD) $(BUILD_EX)
