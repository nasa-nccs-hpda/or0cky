.SUFFIXES:

default: diags

HOMEDIR = $(wildcard ~)
MODELERC ?= $(HOMEDIR)/.modelErc
sinclude $(MODELERC)

export MODEL_E_ROOT:= $(shell pwd)/../..
export CONFIG_DIR := $(MODEL_E_ROOT)/config

include $(CONFIG_DIR)/rules.mk

diags:
	sh ./compscr "$(F90) $(FFLAGS_ALL)" "$(LIBS)"

clean:
	rm prtaj prtareg prtajl prtadiurn prtconsrv prtrvr prtisccp \
	prtolnst prtotj regrid_to_latlon_mod.mod scaleacc agcstat sumfiles \
	write_giss2d write_2d_as_giss4d diffreport prtostat \
	dft_mod.mod prtspeca fuse_ajlagc prtwp
 
