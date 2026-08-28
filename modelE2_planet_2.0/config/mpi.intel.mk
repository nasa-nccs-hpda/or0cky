# This config file assumes that Intel MPI is loaded as a shell module. 
# It may need some updates if one wants to specify its location
# with "MPIDIR".

# if MPIDIR is set these settings may overwrite the onees from the module
# make sure they are compatible
# use this only when needed, i.e. when compiling with gfortran
ifneq (${MPIDIR},)
FFLAGS += -I${MPIDIR}/include
F90FLAGS += -I${MPIDIR}/include
LIBS += -L${MPIDIR}/lib
endif

MPIRUN=$(shell which mpirun)

ifeq ($(MPIRUN),)
  $(error No MPI modules loaded)
endif

VER := $(subst ., ,$(word 8,$(shell $(MPIRUN) --version 2>&1)))
VER_MAJOR := $(word 1,$(VER))
VER_MINOR := $(word 2,$(VER))

ifneq (,$(filter 2019 2021,$(VER_MAJOR)))
LIBS += -lmpifort -lmpi -lrt -lpthread
else
LIBS += -lmpigf -lmpi -lmpigi -ldl -lrt -lpthread
endif

