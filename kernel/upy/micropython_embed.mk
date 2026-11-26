# Generates the MicroPython embed package (ports/embed) for ringzero. Run from kernel/upy (the
# Makefile's `upy` target does it): mpconfigport.h here is picked up through embed.mk's -I.
# MICROPYTHON_TOP and PACKAGE_DIR are passed in by the top-level Makefile.
include $(MICROPYTHON_TOP)/ports/embed/embed.mk
