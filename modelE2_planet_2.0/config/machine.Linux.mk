# Linux - specific options

CPP = /lib/cpp -P -traditional
CPPFLAGS = -DMACHINE_Linux
ECHO_FLAGS = -e

IS_AARCH64 = NO
ifneq (,$(filter aarch64, $(shell uname -a)))
  IS_AARCH64 = YES
endif
