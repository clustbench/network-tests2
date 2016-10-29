# Wrapper for AC_ARG_ENABLE for basic rules
AC_DEFUN([AX_ARG_ENABLE], [
	AC_ARG_ENABLE([$1], [AS_HELP_STRING([--enable-$1], [$4])], [
		AS_CASE(["${enableval}"],
			[yes], [AS_VAR_APPEND([GOALS], [$2])],
			[no],  [],
			[AC_MSG_ERROR([Bad value ${enableval} for --enable-$1])]
		)
	], [
		AS_VAR_APPEND([GOALS], [$3])
	])
])

# Wrapper for AC_CHECK_PROG with gentle error
AC_DEFUN([AX_CHECK_PROG], [
	AC_CHECK_PROG([HAVE_$1], $$1, [true], [false])
	AS_IF([test "x$HAVE_$1" == "xfalse"], [AC_MSG_ERROR([$2])])
])

# Wrapper for AC_CHECK_PROG with gentle error
AC_DEFUN([AX_RAW_CHECK_PROG], [
	AC_CHECK_PROG([HAVE_$1], $1, [true], [false])
	AS_IF([test "x$HAVE_$1" == "xfalse"], [AC_MSG_ERROR([$2])])
])

# Wrapper for AC_CHECK_LIB for list of functions
AC_DEFUN([AX_CHECK_LIB], [
	for sub in $2
	do
		AC_CHECK_LIB([$1], [$sub], [
			AS_IF([echo $LIBS | grep -q "$1"], [], [
				AS_VAR_APPEND([LIBS], [-l$1])
			])
		], [
			AC_MSG_ERROR([Function $sub does not exist in $1])
		])
	done
])

