
F90 = ifx
# include the minor number only in 1 or 2-digit releases
IFX_RELEASE := $(shell ifx --version | perl -e \
  'while(<>){ if(/ifx.* (\d\d?\.\d+|\d+(?=\.))/) { print "$$1"; } }')
FMAKEDEP = $(SCRIPTS_DIR)/sfmakedepend
CMP_MOD = $(SCRIPTS_DIR)/compare_module_file.pl -compiler INTEL-ifort-9-0-on-LINUX

FFLAGS   = -fpp -O2 -convert big_endian -qno-openmp-simd
F90FLAGS = -fpp -O2 -convert big_endian -free -qno-openmp-simd
LFLAGS = -O2
CPPFLAGS += -DCOMPILER_Intel8 -DCONVERT_BIGENDIAN -DCOMPILER_ifx

ifeq ($(MP),YES)
FFLAGS += -fiopenmp
F90FLAGS += -fiopenmp
LFLAGS += -fiopenmp
endif
R8 = -r8
EXTENDED_SOURCE = -extend_source

# two-digit releases should include the minor number
SUPPORTED_RELEASES = 2021 2022

ifneq ($(SKIP_COMPILER_CHECK),YES)
ifeq ($(findstring $(IFX_RELEASE),$(SUPPORTED_RELEASES)),)
  $(error ifx version $(IFX_RELEASE) is not supported by this code. \
          Use one of: $(SUPPORTED_RELEASES) . \
          If you insist on using an unsupported version, you can do it at your \
          own risk by appending "SKIP_COMPILER_CHECK=YES" to the compilation command )
endif
endif


# flags needed for particular releases

# default flags for latest releases (works for 2021):
FFLAGS_RELEASE = -assume protect_parens -fp-model=precise -warn nousage

# if some releases require different flags enter them here
ifeq ($(IFORT_RELEASE),2022)
$(error ifx 2022 was not tested yet)
#FFLAGS_RELEASE = 
endif


FFLAGS += $(FFLAGS_RELEASE)
F90FLAGS += $(FFLAGS_RELEASE)

# these were not yet fixed for ifx
ifeq ($(COMPILE_WITH_TRAPS),YES)
FFLAGS += -CB -fpe0 -ftrapuv -traceback
LFLAGS += -CB -fpe0 -ftrapuv -traceback
F90FLAGS += -CB -fpe0 -ftrapuv -traceback
LFLAGSF += -CB -fpe0 -ftrapuv -traceback
endif
