#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
# Do not forget to update version.
AC_INIT([network-tests2], [2.0.1], [Alexey Salnikov <salnikov@cs.msu.ru>])
AC_PREFIX_DEFAULT([$HOME/network_tests-$PACKAGE_VERSION])

# Directory with user-defined macros (macros with 'AX_' prefix).
AC_CONFIG_MACRO_DIR([ac-macros])

# Configure variables declaration.
# TODO: Change the value for qmake for QT5 later
AC_ARG_VAR([QMAKE], [Place where qmake-qt4 is installed])
AC_ARG_VAR([MPICC], [Path to MPI2 compatible C-compiler])
AC_ARG_VAR([MPICXX], [Path to MPI2 compatible C++-compiler])
AC_ARG_VAR([MPI_CFLAGS], [Flags for MPI C compiler by default same with CFLAGS])
AC_ARG_VAR([MPI_CXXFLAGS], [Flags for MPI C++ compiler by default same with CXXFLAGS])
AC_ARG_VAR([MPI_LIBS], [Additional libs for parallel program])
AC_ARG_VAR([MPI_LDFLAGS], [Path to additional libs for parallel program])
AC_ARG_VAR([GOALS], [List of goals to be built])

# Explicitly enables build for network_viewer_qt_v2 component.
AX_ARG_ENABLE([qt-gui], [" network_viewer_qt_v2 "], [""], [
	Enable Qt version of network-viewer program (disabled by default)
])

# Explicitly enables build for network_test component.
#AX_ARG_ENABLE([network-test], [" network_test "], [" network_test "], [
#	Enable testing (enabled by default)
#])

#Explicitly enables build for network_test component.
AX_ARG_ENABLE([interconnect_benchmark], [" interconnect_benchmark "], [" interconnect_benchmark "], [
	Enable testing (enabled by default)
])

# Explicitly enables build for clustering component.
AX_ARG_ENABLE([clustering], [" clustering "], [""], [
	Enable clustering (disabled by default)
])

#Explicitly enables build for gpu_tests component.
AX_ARG_ENABLE([gpu_tests], [" gpu_tests "], [" gpu_tests_benchmark "], [
	Enable testing (enabled by default)
])

# Enables build for all components of the project.
AC_ARG_ENABLE([all], [AS_HELP_STRING([--enable-all], [Enable all modules])], [
	AS_CASE(["${enableval}"],
		[yes], [AC_SUBST([GOALS], ["interconnect_benchmark network_viewer_qt_v2 clustering gpu_tests"])],
		[no],  [],
		[AC_MSG_ERROR([Bad value ${enableval} for --enable-all])]
	)
])

# Checks for non-default specific tools.
AS_IF([test "x$MPICC" = "x"], [AC_SUBST([MPICC], [mpicc])])
AS_IF([test "x$MPICXX" = "x"], [AC_SUBST([MPICXX], [mpicxx])])
# TODO: Change the value for qmake for QT5 later
AS_IF([test "x$QMAKE" = "x"], [AC_SUBST([QMAKE], [qmake-qt4])])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_LN_S
AC_PROG_RANLIB
AX_CHECK_PROG([MPICC], [MPI compiler for C programming language has not been found])
AX_CHECK_PROG([MPICXX], [MPI compiler for C++ programming language has not been found])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for standart library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([clock_gettime floor gethostname gettimeofday localeconv memset sqrt strchr strstr])

# Checks for required function of netcdf library.
AX_CHECK_LIB([netcdf], [nc_create nc_sync nc_close nc_enddef nc_def_dim \
	nc_def_var nc_put_var_int nc_put_vara_double
])

# Checks for chosen components
for target in $GOALS
do
	AS_CASE(["$target"],
		[*"network_viewer_qt"*], [
			# Checks for tools and libraries required by network_viewer_qt_v2
			AX_CHECK_PROG([QMAKE], [qmake-qt4 is required but has not been found])
			AX_RAW_CHECK_PROG([lrelease], [lrelease is required but has not been found])
			AX_CHECK_LIB([netcdf], [nc_open nc_inq_varid nc_get_var1 \
				nc_inq_nvars nc_inq_varids nc_inq_varndims nc_inq_dimlen \
				nc_inq_vardimid nc_get_vara_double nc_get_var1_int
			])
			# TODO: Add checks for libraries.
			# -lqwt-qt4 -lGLU -lQtOpenGL -lQtGui -lQtCore -lGL -lpthread
		],
		[*"interconnect_benchmark"*], [
			# Checks for tools and libraries required by network_test
			AX_CHECK_LIB([netcdf], [nc_open nc_get_var_int nc_get_vara_double \
				nc_inq_dimid nc_inq_dimlen nc_inq_varid
			])
		]
	)
done

# Use flags of non-MPI compilers for MPI ones
AS_IF([test "x$MPI_CFLAGS" = "x"], [AC_SUBST([MPI_CFLAGS], $CFLAGS)])
AS_IF([test "x$MPI_CXXFLAGS" = "x"], [AC_SUBST([MPI_CXXFLAGS], $CXXFLAGS)])
AS_IF([test "x$MPI_LDFLAGS" = "x"], [AC_SUBST([MPI_LDFLAGS], $LDFLAGS)])
AS_IF([test "x$MPI_LIBS" = "x"], [AC_SUBST([MPI_LIBS], $LIBS)])

# Define output files to be built from templates
AC_CONFIG_FILES([config Makefile clustbench_config.h])
AC_OUTPUT
