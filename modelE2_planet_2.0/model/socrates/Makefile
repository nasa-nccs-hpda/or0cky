# makefile for the component shared
.SUFFIXES:

SUPPORTED_OPTIONS = 

# SOCRATES
include $(SOCRATESPATH)/make/Mk_mod_gencore
include $(SOCRATESPATH)/make/Mk_src_gencore
include $(SOCRATESPATH)/make/Mk_src_radcore
F90SRCS = \
	$(MODULES_GENCORE:.f90=.F90) \
	$(SOURCE90_GENCORE:.f90=.F90) \
	$(SOURCE_RADCORE:.f90=.F90) \
	def_planetstr.F90

SRC_DIRS = interface
ABS_SRC_DIRS = $(SOCRATESPATH)/src/radiance_core

ifndef GISSCLIM_DIR
       GISSCLIM_DIR := $(shell pwd)/../..
endif
ifndef CONFIG_DIR
       CONFIG_DIR := $(GISSCLIM_DIR)/config
endif

include $(CONFIG_DIR)/base.mk
include $(CONFIG_DIR)/rules.mk

