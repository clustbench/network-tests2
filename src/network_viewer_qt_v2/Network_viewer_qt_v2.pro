QT += opengl
CONFIG += qt
TARGET = network_viewer_qt_v2
TEMPLATE = app
HEADERS += core/err_codes.h \
		   core/data_abstract.h \
		   core/data_netcdf.h \
		   core/data_text.h \
		   core/cntrlr_abstract.h \
		   core/cntrlr_single.h \
		   core/cntrlr_deviat.h \
		   core/cntrlr_compare.h \
		   core/matrixraster.h \
		   GUI/err_msgs.h \
		   GUI/tabviewer.h \
		   GUI/matrixviewer.h \
		   GUI/pairviewer.h \
		   GUI/qexpandbox.h \
		   GUI/render_opts.h \
		   GUI/fullviewer.h \
		   GUI/topoviewer.h \
		   GUI/mainwindow.h
SOURCES += main.cpp \
		   core/data_netcdf.cpp \
		   core/data_text.cpp \
		   core/cntrlr_single.cpp \
		   core/cntrlr_deviat.cpp \
		   core/cntrlr_compare.cpp \
		   core/matrixraster.cpp \
		   core/renderer_OpenMP.cpp \
		   GUI/tabviewer.cpp \
		   GUI/matrixviewer.cpp \
		   GUI/qexpandbox.cpp \
		   GUI/render_opts.cpp \
		   GUI/fullviewer.cpp \
		   GUI/topoviewer.cpp \
		   GUI/mainwindow.cpp
FORMS += GUI/mainwindow.ui \
		 GUI/tabviewer.ui \
		 GUI/matrixviewer.ui \
		 GUI/topoviewer.ui
INCLUDEPATH += /usr/include/qwt-qt4
DEPENDPATH += /usr/include/qwt-qt4
LIBS += -L/usr/lib \
		-lqwt-qt4 \
		-lnetcdf \
		-lGLU
RESOURCES += resources.qrc
TRANSLATIONS = translations/nv_tr_ru.ts

# ------------------------------------------
# If these two variables are not empty in 
# '../../config', add new files and folders
# ------------------------------------------
OPENCL_LIB_FLD:OPENCL_INCL {
    SOURCES += core/renderer_OpenCL.cpp
    INCLUDEPATH += $$OPENCL_INCL
    LIBS += -L$$OPENCL_LIB_FLD -lOpenCL
}

