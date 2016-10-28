#!/bin/bash

INSTALL_DIR=$1


if ! [ -d ${INSTALL_DIR} ];
then
	mkdir -p ${INSTALL_DIR};
fi;

if ! [ -d ${INSTALL_DIR}/bin ];
then
	mkdir ${INSTALL_DIR}/bin;
fi;
if ! [ -d ${INSTALL_DIR}/etc ];
then
	mkdir ${INSTALL_DIR}/etc;
fi;
if ! [ -d ${INSTALL_DIR}/var/spool/network_tests2 ];
then
	mkdir -p ${INSTALL_DIR}/var/spool/network_tests2;
fi;
if ! [ -d ${INSTALL_DIR}/lib ];
then
    mkdir ${INSTALL_DIR}/lib;
fi;
if ! [ -d ${INSTALL_DIR}/include ];
then
	mkdir ${INSTALL_DIR}/include;
fi;
if ! [ -d ${INSTALL_DIR}/java ];
	then
	mkdir ${INSTALL_DIR}/java;
fi;
if ! [ -d ${INSTALL_DIR}/doc ];
then
	mkdir ${INSTALL_DIR}/doc;
fi;




exit 0


