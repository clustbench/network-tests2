# File with user-defined macros

# Wrapper for AC_ARG_ENABLE for basic rules
# Arguments:
# $1 - name of the configure option
# $2 - name of corresponding GOAL to be built
# $3 - default value (empty string if GOAL not to be nuilt by default or $2)
# $4 - help string for configure option
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

# Wrapper for AC_CHECK_PROG with configure variable and gentle error
# Arguments:
# $1 - configure variable for corresponding tool
# $2 - error string to be shown if tool doesn't exist
AC_DEFUN([AX_CHECK_PROG], [
	AC_CHECK_PROG([HAVE_$1], $$1, [true], [false])
	AS_IF([test "x$HAVE_$1" == "xfalse"], [AC_MSG_ERROR([$2])])
])

# Wrapper for AC_CHECK_PROG with tool name and gentle error
# Arguments:
# $1 - tool name
# $2 - errorstring to be shown if tool doesn't exist
AC_DEFUN([AX_RAW_CHECK_PROG], [
	AC_CHECK_PROG([HAVE_$1], $1, [true], [false])
	AS_IF([test "x$HAVE_$1" == "xfalse"], [AC_MSG_ERROR([$2])])
])

# Wrapper for AC_CHECK_LIB for list of required functions
# Argumenst:
# $1 - library name (without 'lib' prefix and extension suffix)
# $2 - list of required functions
AC_DEFUN([AX_CHECK_LIB], [
	for sub in $2
	do
		AC_CHECK_LIB([$1], [$sub], [
			# The check below is necessary for deduplication of $1 library
			# within LIBS variable
			AS_IF([echo $LIBS | grep -q "$1"], [], [
				AS_VAR_APPEND([LIBS], [-l$1])
			])
		], [
			AC_MSG_ERROR([Function $sub does not exist in $1])
		])
	done
])
