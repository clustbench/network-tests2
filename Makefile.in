#
# This file is a part of the PARUS project and  makes the core of the parus system
# Copyright (C) 2006  Alexey N. Salnikov (salnikov@cmc.msu.ru)
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

include ./config
SHELL=/bin/bash
GOALS = network_test java network_viewer_qt clustering
#INSTALL_DIR = .

all: core $(GOALS) 

install: core $(GOALS) prepare_install $(INSTALL_DIR)/config install_doc parus_config.h
	$(MAKE) install -C ./src/network_test
	$(MAKE) install -C ./src/core
	$(MAKE) install -C ./src/network_viewer_qt_v2
	$(MAKE) install -C ./src/java
	$(MAKE) install -C ./src/clustering
	echo $(PARUS_VERSION) > $(INSTALL_DIR)/version
	cp ./parus_config.h $(INSTALL_DIR)/include/parus_config.h



uninstall:
	if [ -f $(INSTALL_DIR)/version ]; then	\
	 rm -rf $(INSTALL_DIR);			\
	else					\
	echo "Parus system are not installed";	\
	fi

force_uninstall:
	rm -rf $(INSTALL_DIR)


install_doc:
	cp -rf ./doc/* $(INSTALL_DIR)/doc

core: parus_config.h
	$(MAKE) -C ./src/core

install_core: 
	$(MAKE) install -C ./src/core

network_test:
	$(MAKE) -C ./src/network_test

network_viewer_qt:
	$(MAKE) -C ./src/network_viewer_qt_v2

clustering:
	$(MAKE) -C ./src/clustering

java:
	$(MAKE) -C ./src/java


clean:
	rm -f ./parus_config.h
	$(MAKE) clean -C ./src/network_test
	$(MAKE) clean -C ./src/core
	$(MAKE) clean -C ./src/network_viewer_qt_v2
	$(MAKE) clean -C ./src/java
	$(MAKE) clean -C ./src/clustering


prepare_install:
	./prepare_install.sh $(INSTALL_DIR)

parus_config.h:
	echo "#ifndef __PARUS_CONFIG_H__"			>  parus_config.h
	echo "#define __PARUS_CONFIG_H__"			>> parus_config.h
	echo "	#define  PARUS_VERSION \"$(PARUS_VERSION)\""	>> parus_config.h
	echo "	#define  PARUS_INSTALL_DIR \"$(INSTALL_DIR)\""	>> parus_config.h
	echo "  #define  PARUS_DATA_DIR \"$(INSTALL_DIR)/var/spool/network_tests2\"" >> parus_config.h
	echo "#endif /* __PARUS_CONFIG_H__ */" 			>> parus_config.h


$(INSTALL_DIR)/config:
	echo "PARUS_VERSION=$(PARUS_VERSION)"		>  $(INSTALL_DIR)/config
	echo "INSTALL_DIR=$(INSTALL_DIR)"		>> $(INSTALL_DIR)/config
	echo "MPI_HOME=$(MPI_HOME)"			>> $(INSTALL_DIR)/config
	echo "MPI_cc=$(MPI_cc)"				>> $(INSTALL_DIR)/config
	echo "MPI_CC=$(MPI_CC)"				>> $(INSTALL_DIR)/config
	echo "MPI_CLINKER=$(MPI_CLINKER)"		>> $(INSTALL_DIR)/config
	echo "MPI_CCLINKER=$(MPI_CCLINKER)"		>> $(INSTALL_DIR)/config
	echo "MPI_LIB_PATH=$(MPI_LIB_PATH)"		>> $(INSTALL_DIR)/config
	echo "MPI_LIBS=$(MPI_LIBS)"			>> $(INSTALL_DIR)/config 
	echo "MPI_cc_INCLUDE=$(MPI_cc_INCLUDE) -I $(INSTALL_DIR) -I $(INSTALL_DIR)/include" >> $(INSTALL_DIR)/config
	echo "MPI_CC_INCLUDE=$(MPI_CC_INCLUDE) -I $(INSTALL_DIR) -I $(INSTALL_DIR)/include" >> $(INSTALL_DIR)/config
	echo "MPI_cc_FLAGS=$(MPI_cc_FLAGS)"		>> $(INSTALL_DIR)/config
	echo "MPI_CC_FLAGS=$(MPI_CC_FLAGS)"		>> $(INSTALL_DIR)/config 
	echo "MPI_CLINKER_FLAGS=$(MPI_CLINKER_FLAGS)"	>> $(INSTALL_DIR)/config
	echo "MPI_CCLINKER_FLAGS=$(MPI_CCLINKER_FLAGS)"	>> $(INSTALL_DIR)/config
	echo "CC=$(CC)"					>> $(INSTALL_DIR)/config
	echo "CCC=$(CCC)"				>> $(INSTALL_DIR)/config
	echo "CC_FLAGS=$(CC_FLAGS)"			>> $(INSTALL_DIR)/config
	echo "CCC_FLAGS=$(CCC_FLAGS)"			>> $(INSTALL_DIR)/config
	echo "CC_INCLUDE=$(CC_INCLUDE)   -I $(INSTALL_DIR) -I $(INSTALL_DIR)/include" >> $(INSTALL_DIR)/config
	echo "CCC_INCLUDE=$(CCC_INCLUDE) -I $(INSTALL_DIR) -I $(INSTALL_DIR)/include" >> $(INSTALL_DIR)/config
	echo "CLINKER=$(CLINKER)"			>> $(INSTALL_DIR)/config
	echo "CCLINKER=$(CCLINKER)"			>> $(INSTALL_DIR)/config
	echo "CLINKER_FLAGS=$(CLINKER_FLAGS)"		>> $(INSTALL_DIR)/config
	echo "CCLINKER_FLAGS=$(CCLINKER_FLAGS)"		>> $(INSTALL_DIR)/config
	echo "LIB_PATH=$(LIB_PATH)"			>> $(INSTALL_DIR)/config
	echo "LIBS=$(LIBS)"				>> $(INSTALL_DIR)/config
	echo "ANT=$(ANT)"				>> $(INSTALL_DIR)/config
	echo "# full path to 'CL/opencl.h' (on Windows/UNIX) or 'OpenCL/opencl.h' (on Apple)" >> $(INSTALL_DIR)/config
	echo "# (WITHOUT 'CL/opencl.h' or 'OpenCL/opencl.h'!)" 		>> $(INSTALL_DIR)/config
	echo "OPENCL_INCL=$(OPENCL_INCL)"				>> $(INSTALL_DIR)/config
	echo "# full path to OpenCL library (for example, '/usr/lib')" 	>> $(INSTALL_DIR)/config
	echo "OPENCL_LIB_FLD=$(OPENCL_LIB_FLD)"				>> $(INSTALL_DIR)/config

.PHONY: src/java
