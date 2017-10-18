BUILD = build
SRC = src

INSTALL_PREFIX ?= $(HOME)/software
MAGMADIR    ?= $(INSTALL_PREFIX)/magma-2.2.0
OPENBLASDIR ?= $(INSTALL_PREFIX)/OpenBLAS-0.2.19
MLEARNDIR   ?= $(INSTALL_PREFIX)/libmlearn

all: mlearn

install-deps:
	@echo -e "\nBuilding OpenBLAS...\n"

	mkdir -p $(OPENBLASDIR)
	wget -q https://github.com/xianyi/OpenBLAS/archive/v0.2.19.tar.gz
	tar -xf v0.2.19.tar.gz

	cd OpenBLAS-0.2.19 && cp lapack-netlib/make.inc.example lapack-netlib/make.inc
	+$(MAKE) -s -C OpenBLAS-0.2.19 NO_LAPACK=0 TARGET=NEHALEM
	+$(MAKE) -s -C OpenBLAS-0.2.19 install PREFIX=$(OPENBLASDIR)

	rm -rf v0.2.19.tar.gz OpenBLAS-0.2.19

	@echo -e "\nBuilding MAGMA...\n"

	mkdir -p $(MAGMADIR)
	wget -q http://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-2.2.0.tar.gz
	tar -xf magma-2.2.0.tar.gz

	cd magma-2.2.0 && cp make.inc-examples/make.inc.openblas make.inc
	+$(MAKE) -s -C magma-2.2.0
	+$(MAKE) -s -C magma-2.2.0 install prefix=$(MAGMADIR)

	rm -rf magma-2.2.0.tar.gz magma-2.2.0

	@echo -e "\nInstallation complete.\n"

$(BUILD):
	mkdir -p $(BUILD)

mlearn: $(SRC)/**/*.h $(SRC)/**/*.cpp | $(BUILD)
	cd $(BUILD) && cmake .. -DCMAKE_INSTALL_PREFIX=$(INSTALL_PREFIX)/libmlearn
	+$(MAKE) -C $(BUILD) install

clean:
	rm -rf $(BUILD)
