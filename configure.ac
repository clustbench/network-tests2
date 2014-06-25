#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([network-tests2], [2.0.1], [Alexey Salnikov <salnikov@cs.msu.ru>])
#AC_CONFIG_SRCDIR([parus_config.h])
#AC_CONFIG_HEADERS([config.h])

AC_ARG_VAR([JAVAC],[Place where javac is installed])
AC_ARG_VAR([JAR],[Place where jar is installed])
AC_ARG_VAR([ANT],[Place where ant utility is installed])

AC_ARG_VAR([MPICC],[Path to MPI2 compatible C-compiler])
AC_ARG_VAR([MPICXX],[Path to MPI2 compatible C++-compiler])
AC_ARG_VAR([MPI_CFLAGS],[Flags for MPI C compiler by default same with CFLAGS])
AC_ARG_VAR([MPI_CXXFLAGS],[Flags for MPI C++ compiler by default same with CXXFLAGS])
AC_ARG_VAR([MPI_LIBS],[Additional libs for parallel program])
AC_ARG_VAR([MPI_LDFLAGS],[Path to additional libs for parallel program])

AC_ARG_ENABLE([java-gui],
              [AS_HELP_STRING([--enable-java-gui],[Enable Java version of network-viewer program. It is disabled by default.])],
              [JAVA_GUI=disabled],
              [JAVA_GUI=enabled]
             )


if test "x$JAVAC" = "x" ; then
    AC_SUBST([JAVAC],[javac])
fi

if test "x$JAR" = "x" ; then
    AC_SUBST([JAR],[jar])
fi

if test "x$ANT" = "x" ; then
    AC_SUBST([ANT],[ant])
fi


if test "x$MPICC" = "x" ; then
    AC_SUBST([MPICC],[mpicc])
fi

if test "x$MPICXX" = "x" ; then
    AC_SUBST([MPICXX],[mpicxx])
fi


AC_PREFIX_DEFAULT([$HOME/network_tests-$PACKAGE_VERSION])




# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
#AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_RANLIB

AC_CHECK_PROG([HAVE_MPICC],$MPICC, [true], [false])
if test "x$HAVE_MPICC" = "xfalse" ; then
    AC_MSG_ERROR([MPI compiler for C programming language is not found])
fi

AC_CHECK_PROG([HAVE_MPICXX],$MPICXX, [true], [false])
if test "x$HAVE_MPICXX" = "xfalse" ; then
    AC_MSG_ERROR([MPI compiler for C++ programming language is not found])
fi

if test "x$MPI_CFLAGS" = "x" ; then
    AC_SUBST([MPI_CFLAGS],$CFLAGS)
fi

if test "x$MPI_CXXFLAGS" = "x" ; then
    AC_SUBST([MPI_CXXFLAGS],$CXXFLAGS)
fi

if test "x$MPI_LDFLAGS" = "x" ; then
    AC_SUBST([MPI_LDFLAGS],$LDFLAGS)
fi

if test "x$MPI_LIBS" = "x" ; then
    AC_SUBST([MPI_LIBS],$LIBS)
fi




#AX_MPI([`echo MPI found'],[`echo MPI not FOUND'])
#AC_OPENMP
# Checks for libraries.
# FIXME: Replace `main' with a function in `-lnetcdf':
AC_CHECK_LIB([netcdf], [nc_open])
# FIXME: Replace `main' with a function in `-lparus_network':
#AC_CHECK_LIB([parus_network], [main])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([clock_gettime floor gethostname gettimeofday localeconv memset sqrt strchr strstr])

AC_CONFIG_FILES([config Makefile])
AC_OUTPUT
