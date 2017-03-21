NTESTS_ROOT = $(shell pwd)

###############################
##### PREPARE DIRECTORIES #####
###############################

PREFIX := ${HOME}/_scratch/network_tests_prefix
${PREFIX}:
	mkdir -p $@

export PATH := ${PREFIX}/bin:${PATH}

DOWNLOADS_DIR := ${NTESTS_ROOT}/downloads
${DOWNLOADS_DIR}:
	mkdir -p $@

DISTROS_DIR := ${NTESTS_ROOT}/distros
${DISTROS_DIR}:
	mkdir -p $@

# directory where all directories for compiling are
BUILDS_ROOT_DIR = ${NTESTS_ROOT}/builds
${BUILDS_ROOT_DIR}:
	mkdir -p $@

# directory for phony files which represent targets
TARGETS_DIR = ${NTESTS_ROOT}/make_targets
${TARGETS_DIR}:
	mkdir -p $@

###################################
##### END PREPARE DIRECTORIES #####
###################################

##################
###### CURL ######
##################

CURL_FILENAME = curl-7.50.3.zip
CURL_URL = https://curl.haxx.se/download/${CURL_FILENAME}

# this is the name of directory inside curl archive
CURL_DIR = curl-7.50.3

# where the downloaded archive will be saved
CURL_ARCHIVE_PATH = ${DOWNLOADS_DIR}/${CURL_FILENAME}

# where curl will be extracted from archive
CURL_EXTRACTED_PATH = ${DISTROS_DIR}/${CURL_DIR}

# a file we touch when extraction is successful
CURL_EXTRACTED_SUCCESSFULLY = ${TARGETS_DIR}/curl_extracted_successfully



# download curl
${CURL_ARCHIVE_PATH}: | ${DOWNLOADS_DIR}
	wget -q --output-document $@ ${CURL_URL}


# extract curl from the archive
${CURL_EXTRACTED_SUCCESSFULLY}: ${CURL_ARCHIVE_PATH} | ${DISTROS_DIR} ${TARGETS_DIR}
	unzip -o -q ${CURL_ARCHIVE_PATH} -d ${DISTROS_DIR}
	touch $@

# directory where curl is built
CURL_BUILD_DIR = ${BUILDS_ROOT_DIR}/${CURL_DIR}
${CURL_BUILD_DIR}:
	mkdir -p $@


# build curl
CURL_BUILT_SUCCESSFULLY = ${TARGETS_DIR}/curl_built_successfully
${CURL_BUILT_SUCCESSFULLY}: ${CURL_EXTRACTED_SUCCESSFULLY} | ${PREFIX} ${TARGETS_DIR} ${CURL_BUILD_DIR}
	cd ${CURL_BUILD_DIR} && \
	  ${CURL_EXTRACTED_PATH}/configure --prefix=${PREFIX} --exec-prefix=${PREFIX} && \
	  $(MAKE)
	touch $@


# install curl to prefix
CURL_INSTALLED = ${TARGETS_DIR}/curl_installed
${CURL_INSTALLED}: ${CURL_BUILT_SUCCESSFULLY} | ${PREFIX} ${TARGETS_DIR}
	$(MAKE) -C ${CURL_BUILD_DIR} install
	touch $@

######################
###### END CURL ######
######################



##################
###### HDF5 ######
##################


HDF5_DIR = hdf5-1.10.0-patch1-linux-centos7-x86_64-gcc485-shared
HDF5_EXTRACTED_PATH = ${DISTROS_DIR}/${HDF5_DIR}


# download hdf5 archive
HDF5_FILENAME = ${HDF5_DIR}.tar.gz
HDF5_URL = https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.0-patch1/bin/linux-centos7-x86_64-gcc485/${HDF5_FILENAME}
HDF5_ARCHIVE_PATH = ${DOWNLOADS_DIR}/${HDF5_FILENAME}
${HDF5_ARCHIVE_PATH}: | ${DOWNLOADS_DIR}
	wget -q --output-document $@ ${HDF5_URL}


# extract hdf5 archive
HDF5_EXTRACTED_SUCCESSFULLY = ${TARGETS_DIR}/hdf5_extracted_successfully
${HDF5_EXTRACTED_SUCCESSFULLY}: ${HDF5_ARCHIVE_PATH} | ${DISTROS_DIR} ${TARGETS_DIR}
	tar -xzf ${HDF5_ARCHIVE_PATH} -C ${DISTROS_DIR}
	touch $@


# copy hdf5 files into prefix
HDF5_INSTALLED_SUCCESSFULLY = ${TARGETS_DIR}/hdf5_installed_successfully
${HDF5_INSTALLED_SUCCESSFULLY}: ${HDF5_EXTRACTED_SUCCESSFULLY} | ${PREFIX} ${TARGETS_DIR}
	# copy all subdirectories to prefix
	find ${HDF5_EXTRACTED_PATH} -mindepth 1 -maxdepth 1 -type d -execdir cp -R {} ${PREFIX} \;
	touch $@

######################
###### END HDF5 ######
######################



###################
###### CMAKE ######
###################

# download cmake
CMAKE_DIR = cmake-3.6.2-Linux-x86_64
CMAKE_FILENAME = ${CMAKE_DIR}.tar.gz
CMAKE_URL = https://cmake.org/files/v3.6/${CMAKE_FILENAME}
CMAKE_ARCHIVE_PATH = ${DOWNLOADS_DIR}/${CMAKE_FILENAME}
${CMAKE_ARCHIVE_PATH}: | ${DOWNLOADS_DIR}
	wget -q --output-document $@ ${CMAKE_URL}


# extract cmake archive
CMAKE_EXTRACTED_SUCCESSFULLY = ${TARGETS_DIR}/cmake_extracted_successfully
${CMAKE_EXTRACTED_SUCCESSFULLY}: ${CMAKE_ARCHIVE_PATH} | ${DISTROS_DIR} ${TARGETS_DIR}
	tar -xzf ${CMAKE_ARCHIVE_PATH} -C ${DISTROS_DIR}
	touch $@


# copy cmake files into prefix
CMAKE_EXTRACTED_PATH = ${DISTROS_DIR}/${CMAKE_DIR}
CMAKE_INSTALLED_SUCCESSFULLY = ${TARGETS_DIR}/cmake_installed_successfully
${CMAKE_INSTALLED_SUCCESSFULLY}: ${CMAKE_EXTRACTED_SUCCESSFULLY} | ${PREFIX} ${TARGETS_DIR}
	# copy all subdirectories to prefix
	find ${CMAKE_EXTRACTED_PATH} -mindepth 1 -maxdepth 1 -type d -execdir cp -R {} ${PREFIX} \;
	touch $@

#######################
###### END CMAKE ######
#######################




####################
###### NETCDF ######
####################

NETCDF_DIR = netcdf-4.4.1



# download netcdf
NETCDF_FILENAME = ${NETCDF_DIR}.zip
NETCDF_URL = ftp://ftp.unidata.ucar.edu/pub/netcdf/${NETCDF_FILENAME}
NETCDF_ARCHIVE_PATH = ${DOWNLOADS_DIR}/${NETCDF_FILENAME}
${NETCDF_ARCHIVE_PATH}: | ${DOWNLOADS_DIR}
	wget -q --output-document $@ ${NETCDF_URL}



# extract netcdf
NETCDF_EXTRACTED_SUCCESSFULLY = ${TARGETS_DIR}/netcdf_extracted_successfully
${NETCDF_EXTRACTED_SUCCESSFULLY}: ${NETCDF_ARCHIVE_PATH} | ${DISTROS_DIR} ${TARGETS_DIR}
	unzip -o -q ${NETCDF_ARCHIVE_PATH} -d ${DISTROS_DIR}
	touch $@

NETCDF_EXTRACTED_PATH = ${DISTROS_DIR}/${NETCDF_DIR}


# create directory for building netcdf
NETCDF_BUILD_DIR = ${BUILDS_ROOT_DIR}/${NETCDF_DIR}
${NETCDF_BUILD_DIR}:
	mkdir -p $@


# build netcdf
NETCDF_BUILT_SUCCESSFULLY = ${TARGETS_DIR}/netcdf_built_successfully
${NETCDF_BUILT_SUCCESSFULLY}: ${NETCDF_EXTRACTED_SUCCESSFULLY} ${HDF5_INSTALLED_SUCCESSFULLY} ${CURL_INSTALLED}| ${PREFIX} ${TARGETS_DIR} ${NETCDF_BUILD_DIR}
	cd ${NETCDF_BUILD_DIR} && \
	  LDFLAGS="${LDFLAGS} -L${PREFIX}/lib" CFLAGS="${CFLAGS} -I${PREFIX}/include" CPPFLAGS="${CPPFLAGS} -I${PREFIX}/include" ${NETCDF_EXTRACTED_PATH}/configure --prefix=${PREFIX} --exec-prefix=${PREFIX} && \
	  $(MAKE)
	touch $@


# install netcdf
NETCDF_INSTALLED_SUCCESSFULLY = ${TARGETS_DIR}/netcdf_installed_successfully
${NETCDF_INSTALLED_SUCCESSFULLY}: ${NETCDF_BUILT_SUCCESSFULLY} | ${PREFIX}
	$(MAKE) -C ${NETCDF_BUILD_DIR} install
	touch $@


########################
###### END NETCDF ######
########################


#############################
###### NETWORK TESTS 2 ######
#############################

NETWORK_TESTS2_GIT_REPO ?= https://github.com/clustbench/network-tests2.git
NETWORK_TESTS2_BRANCH ?= master

NETWORK_TESTS2_DIR = network-tests2

# download network_tests2
NETWORK_TESTS2_DISTRO = ${DISTROS_DIR}/${NETWORK_TESTS2_DIR}
${NETWORK_TESTS2_DISTRO}: | ${DISTROS_DIR}
	git clone ${NETWORK_TESTS2_GIT_REPO} ${NETWORK_TESTS2_DISTRO}


# pseudo-prefix to install network-tests2 there
# and then copy all files and directories from there to the actual prefix
# recursively
NETWORK_TESTS2_PREFIX = ${BUILDS_ROOT_DIR}/network-tests2-prefix


# build
NETWORK_TESTS2_BUILD_DIR = ${BUILDS_ROOT_DIR}/${NETWORK_TESTS2_DIR}
NETWORK_TESTS2_BUILT = ${TARGETS_DIR}/network_tests2_built
${NETWORK_TESTS2_BUILT}: ${NETWORK_TESTS2_DISTRO} ${NETCDF_INSTALLED_SUCCESSFULLY} ${HDF5_INSTALLED_SUCCESSFULLY} | ${BUILDS_ROOT_DIR}
	cp -R ${NETWORK_TESTS2_DISTRO} ${BUILDS_ROOT_DIR}
	cd ${NETWORK_TESTS2_BUILD_DIR} && \
	  git checkout origin/${NETWORK_TESTS2_BRANCH} && \
	  ./make_configure.sh && \
	  CFLAGS="${CFLAGS} -I${PREFIX}/include" LDFLAGS="-L${PREFIX}/lib" ./configure --prefix="${NETWORK_TESTS2_PREFIX}" && \
	  $(MAKE) all
	touch $@

# install
NETWORK_TESTS2_INSTALLED = ${TARGETS_DIR}/network_tests2_installed
${NETWORK_TESTS2_INSTALLED}: ${NETWORK_TESTS2_BUILT} | ${PREFIX}
	# pseudo-install to special prefix dir
	# -j1 is used because network-tests2's Makefile is badly written
	# and might not work with number of jobs > 1
	$(MAKE) -C ${NETWORK_TESTS2_BUILD_DIR} -j1 install  # will pseudo-install to special prefix dir
	# now we need to copy it to the actual prefix
	cp -R ${NETWORK_TESTS2_PREFIX}/* ${PREFIX}/
	touch $@

#################################
###### END NETWORK TESTS 2 ######
#################################


#####################
###### CLEANUP ######
#####################

clean_distros:
	rm -rf ${DISTROS_DIR}

clean_prefix:
	rm -rf ${PREFIX}

clean_targets:
	rm -rf ${TARGETS_DIR}

clean_downloads:
	rm -rf ${DOWNLOADS_DIR}

clean_builds:
	rm -rf ${BUILDS_ROOT_DIR}

clean_run_dir:
	rm -rf ${RUN_DIR}

clean: clean_distros clean_prefix clean_targets clean_downloads clean_builds clean_run_dir


#########################
###### END CLEANUP ######
#########################

# Phony targets, meaning they are not supposed to produce any files
.PHONY: print_variables, clean, clean_distros, clean_prefix, clean_targets, clean_downloads, clean_builds, all, run, clean_run_dir


print_variables:
	$(foreach v, $(.VARIABLES), $(info $(v) = $($(v))))


all: ${NETWORK_TESTS2_INSTALLED}


RUN_DIR = ${PREFIX}/run_results
NUM_NODES = 10
run: ${NETWORK_TESTS2_INSTALLED}
	rm -rf ${RUN_DIR}
	mkdir -p ${RUN_DIR}
	cd ${RUN_DIR} && \
	  PATH="${PREFIX}/bin:${PATH}" MANPATH="${PREFIX}/share/man:${MANPATH}" \
	    LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH}" \
	    sbatch -n ${NUM_NODES} --ntasks-per-node 1 ompi ${PREFIX}/bin/network_test2
