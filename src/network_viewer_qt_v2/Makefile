#
# This file is a part of the PARUS project and  makes the core of the parus system
# Copyright (C) 2006  Alexey N. Salnikov
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#
# Alexey N. Salnikov (salnikov@cmc.msu.ru)
#
# 
#  This file makes the nework_viewer_qt program for PARUS instrumental 
#  system.
# 


include ../../config

EXECS= network_viewer_qt_v2

#QMAKE_OPTIONS:= 'CONFIG+=debug'
QMAKE_OPTIONS:=

# making with OpenMP support
QMAKE_OPTIONS += QMAKE_LFLAGS+=-fopenmp QMAKE_CXXFLAGS+=-fopenmp

# making with OpenCL support
ifneq ($(OPENCL_INCL)$(OPENCL_LIB_FLD),$(OPENCL_LIB_FLD)$(OPENCL_INCL))
	QMAKE_OPTIONS += CONFIG+=OPENCL_LIB_FLD OPENCL_LIB_FLD=$(OPENCL_LIB_FLD)
	QMAKE_OPTIONS += CONFIG+=OPENCL_INCL OPENCL_INCL=$(OPENCL_INCL)
endif
	

all: prepare $(EXECS)


prepare:
	@echo "/* This file was generated automatically by 'make' */" >  core/opencl_defs.h
	@echo ""                                                      >> core/opencl_defs.h
ifneq ($(OPENCL_INCL)$(OPENCL_LIB_FLD),$(OPENCL_LIB_FLD)$(OPENCL_INCL))
	@echo "#pragma once"                                          >> core/opencl_defs.h
	@echo ""                                                      >> core/opencl_defs.h
	@echo "#ifndef _OPENCL"                                       >> core/opencl_defs.h
	@echo "  #define _OPENCL 1"                                   >> core/opencl_defs.h
	@echo "#endif"                                                >> core/opencl_defs.h
	@echo "#ifdef __APPLE__"                                      >> core/opencl_defs.h
	@echo "  #include <$(OPENCL_INCL)/OpenCL/opencl.h>"           >> core/opencl_defs.h
	@echo "#else"                                                 >> core/opencl_defs.h
	@echo "  #include <$(OPENCL_INCL)/CL/opencl.h>"               >> core/opencl_defs.h
	@echo "#endif"                                                >> core/opencl_defs.h
endif
	
network_viewer_qt_v2: translations Makefile_network_viewer_qt.mk 
	$(MAKE) -f Makefile_network_viewer_qt.mk

translations:
	$(MAKE) -C translations

Makefile_network_viewer_qt.mk:
	qmake-qt4 $(QMAKE_OPTIONS) -o $@


install: $(EXECS)
	cp -rf $^ $(INSTALL_DIR)/bin

clean:
	@if [ -f ./Makefile_network_viewer_qt.mk ]; then \
		$(MAKE) -f Makefile_network_viewer_qt.mk compiler_clean; \
	fi
	rm -f $(EXECS) Makefile_network_viewer_qt.mk *.o core/opencl_defs.h
	$(MAKE) clean -C translations

# for minimal remake
quick_clean:
	rm -f $(EXECS)


.PHONY: clean prepare translations
