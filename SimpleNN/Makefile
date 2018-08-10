#set source and target for device

EXTRA ?= normal

PROJECT ?= test
BUILDTIME ?= 0
BOARD ?= p510t_sch_ax115

BUILDDIR = $(abspath /home/erwin/build_cache)
SRCDIR = $(abspath ./device)
OUTPUTDIR = $(abspath ./kernels)

#select the host archecture  x86|arm32
PLATFORM = x86
#optinal flows are hw|hw_emu|sw_emu
FLOW        ?= sw_emu
EXTRA_OCCFLAGS  ?= -O3 -fp-relaxed

KERNEL_SRCS = $(SRCDIR)/$(PROJECT).cl

KERNEL_NAME = kernel_$(PROJECT)_$(EXTRA)_$(FLOW)_$(BUILDTIME)
KERNEL_DEFS =
KERNEL_INCS =
KERNEL_DEBUG ?= 0

#select the device binary name
DEV_DIR_TMP = $(BUILDDIR)/$(KERNEL_NAME)
DEV_EXE_TMP = $(DEV_DIR_TMP).aocx
DEV_EXE_META_TMP = $(DEV_DIR_TMP).aoco

DEV_EXE_REPORT = $(DEV_DIR_TMP)/reports

DEV_DIR = $(OUTPUTDIR)/$(KERNEL_NAME)
DEV_EXE = $(DEV_DIR).aocx
DEV_EXE_META = $(DEV_DIR)/.aoco

CXX := g++

CXXFLAGS = -g -Wall -std=c++11

#select whether manually launch free-run kernels
ifeq ($(FLOW),sw_emu)
CXXFLAGS += -DSW_EMU
endif

COMP_CONFIG = $(shell aocl compile-config) -I$(HOST_INCS) -DFPGA_DEVICE
LINK_CONFIG = $(shell aocl link-config)

OCC = aoc
ifeq ($(FLOW),sw_emu)
	OCCFLAGS = -v -report -march=emulator -board=$(BOARD) $(EXTRA_OCCFLAGS)
	REPORTOCCFLAGS = -v -report -march=emulator -board=$(BOARD) $(EXTRA_OCCFLAGS) -c
else ifeq ($(FLOW),hw)
	OCCFLAGS = -v -report -board=$(BOARD) $(EXTRA_OCCFLAGS)
	#OCCFLAGS = -v -report -board $(BOARD)
	REPORTOCCFLAGS = -v -report -board=$(BOARD) $(EXTRA_OCCFLAGS) -c
endif

#debug option
ifeq ($(KERNEL_DEBUG),1)
	OCCFLAGS += -g
endif

.PHONY: all
all: report fpga

.PHONY: fpga
fpga: $(BUILDDIR) $(OUTPUTDIR) $(DEV_EXE)
	@echo Generated $(DEV_EXE)

#TODO Notifications

.PHONY: report
report: $(BUILDDIR) $(DEV_EXE_REPORT)	

$(DEV_EXE): $(DEV_EXE_TMP)

$(DEV_EXE_META): $(DEV_EXE_META_TMP)

$(DEV_EXE_TMP): $(DEV_EXE_REPORT)
	@echo Running full flow for $(patsubst %.aocx,%.aoco,$@) 
	$(OCC) $(OCCFLAGS) $(patsubst %.aocx,%.aoco,$@) -o $@

$(DEV_EXE_REPORT): $(KERNEL_SRCS)
	@echo Generating report for $(patsubst %/,%,$(dir $@))
	$(OCC) $(REPORTOCCFLAGS) $< -o $(patsubst %/,%,$(dir $@))

$(OUTPUTDIR)/%:
	@echo Copying file $< to $@
	cp -f $< $@

$(OUTPUTDIR):
	mkdir -p $@

$(BUILDDIR):
	mkdir -p $@


.PHONY: clean
clean:
	@echo Please clean manually, build dir is $(BUILDDIR), output dir is $(OUTPUTDIR).

.PHONY:
get_build_dir:
	@echo $(DEV_DIR_TMP)

# TODO fix
#rm -rf *.xclbin *.rpt *.xtxt *.json *.dir $(DEV_EXE) $(DEV_EXE_META) *.exe xocc* sdaccel* ./device/*.ll ../common/*.o *.log *.jou *.mon