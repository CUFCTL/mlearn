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
