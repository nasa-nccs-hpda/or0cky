# makefile for the component shared
.SUFFIXES:

SUPPORTED_OPTIONS = 

F90SRCS = apex_interface.F90 apex.F90

ifndef GISSCLIM_DIR
       GISSCLIM_DIR := $(shell pwd)/../..
endif
ifndef CONFIG_DIR
       CONFIG_DIR := $(GISSCLIM_DIR)/config
endif

include $(CONFIG_DIR)/base.mk
include $(CONFIG_DIR)/rules.mk
