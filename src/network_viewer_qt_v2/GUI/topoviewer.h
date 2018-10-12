/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2013  Alexey N. Salnikov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Alexey N. Salnikov salnikov@cmc.msu.ru
 * Pavel S. Bannikov  pashokkk@bk.ru
 */

#pragma once

#include "../core/data_abstract.h"
#include "mainwindow.h"
#include <cstdio>
#include <QGLWidget>
#include <QDialog>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QDebug>
#include <QSpinBox>
class QKeyEvent;
class QMouseEvent;

/* A widget for viewing resulting topology using OpenGL.
   It is used only by TopologyViewer class (defined below).
   You should consider TVWidget class and
   TopologyViewer class as a single class  */
class TVWidget: public QGLWidget {
    Q_OBJECT

	friend class TopologyViewer;
	
  private:
  	  unsigned int x_num; // both width and height of 2D matrices in input file(s)
      unsigned int z_num; // number of 2D matrices in input file(s)
      // both 'x_num' and 'z_num' are defined only once (while reading input file(s))

      unsigned int *edge_counts; // the topology (general graph); it looks like adjacency matrix,
      							 // however each element stores not 0 or 1 but an integer between 0 and 'z_num';
      							 // it allows to represent both existing edges and the "probabilities" of edges;
      							 // its size is x_num*x_num;

      /* visualization */
      float *points_x,*points_y,*points_z; // coordinates of graph's vertices in 3D (calculated approximately!)
      float geom_c_z; // z-coordinate of geometric centre of the graph
      float shift_x,shift_y,shift_z; // store all translations of the graph
      float alpha,beta; // store all rotations of the graph (in OXZ and OYZ correspondently); in degrees
      static const float backgr_clr; // color for widget's background
      QString *host_names; // array of hosts' names (its size is equal to 'x_num')
      unsigned int min_edg_count; // if edge_counts(i,j) is less than 'min_edg_count' then
      							  // do NOT draw an edge between vertices i and j
      bool show_host_names; // determines if hosts' names are drawn near the vertices
      /* visualization: "ideal" topology */
      static const GLfloat ignore_clr; // "ignore" color
      quint8 *i_v_color; // color index of a vertex in "ideal" topology: 0 - 'ignore_clr', 1 - magenta;
      					 // each element handles 8 vertices (| 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |);
      					 // only matching vertices in "real" and "ideal" topology have magenta color
      quint8 *i_e_color; // color index of existing edges in "ideal" topology:
      					 // 0 - 'ignore_clr', 1 - red spectrum, 2 - blue spectrum;
      					 // each element handles 4 edges (| 33 | 33 | 11 | 00 |);
      					 // must be iterated simultaneously with 'edge_counts'!
      unsigned char *i_e_color_val; // real colors of existing edges in "ideal" topology;
      								// values in 'i_e_color_val' correspond to values in 'i_e_color'
      								// so 'i_e_color_val' must be iterated simultaneously with 'i_e_color'

      /* mouse tracker */
      bool mouse_pressed; // 'true' while mouse button is pressed
      bool click_disabled; // 'true' if mouse was moved while its button is pressed
      int x_move,y_move; // mouse movement while its button is pressed
      /* save picture parametrs and button*/
      QPushButton *save_menu_btn; // button for save image as png
      QSpinBox    *save_width ; // width in pixels for image
      QSpinBox    *save_heigth ; // heigth in pixels for image

  private:
      Q_DISABLE_COPY(TVWidget)

      // the only constructor
  	  TVWidget (void);
  	
  	  // destructor
  	  ~TVWidget (void);
  	
  	  // computes coordinates of graph's vertices in 3D space trying to preserve lengths of edges
  	  // (the coordinates are stored in 'points_x', 'points_y' and 'points_z' arrays)
  	  //
  	  // 'matr' - lengths of edges; must have at least x_num*x_num elements
  	  //          ATTENTION: values in 'matr' will be changed!
  	  // 'm_d_impact' - TopologyViewer's variable of the same name
  	  // 'edg_num' - number of all edges
  	  // 'edg50_num' - number of edges with length error not less than 50%
  	  // 'edg99_num' - number of edges with length error not less than 99%
  	  //
      // returns 'false' if there was not enough memory
      bool MapGraphInto3D (double *matr, const double m_d_impact,
      					   unsigned int &edg_num, unsigned int &edg50_num, unsigned int &edg99_num);
  	
  	  void ApplyTransform (void);

      // draws a cone around Oz axis; wide "bottom" is on OXY plane,
      // and "top" is directed towards +Z
      //
      // 'radius' - radius of the cone (can be negative);
      // 'height' - height of the cone (can be negative);
      //			the top point of the cone is placed at (0,0,height)
      // 'slices' - number of points in a polygon which approximate a circle;
      //			it must not be less than 2
      static void tvCone (const float radius, const float height, const unsigned int slices);

  protected:
	  virtual void initializeGL ();
  	
	  virtual void paintGL ();
	
	  virtual void resizeGL (int, int);
	
	  // processes keyboard
      virtual void keyPressEvent (QKeyEvent*);

      // processes left mouse button's presses:
      // initializes mouse position to rotate the graph (in future)
      virtual void mousePressEvent (QMouseEvent*);

      // processes mouse movements: gets mouse movement vector to compute rotation angles
      virtual void mouseMoveEvent (QMouseEvent*);

      // processes left mouse button's releases
      virtual void mouseReleaseEvent (QMouseEvent*);
  private Q_SLOTS:
    void SaveImageMenu ();

    void SaveImage (void) {
        QString fileName;

        fileName = QFileDialog::getSaveFileName(this, tr("Name of file for saving"), QString(),"Graphic files (*.png )");

        if ( !fileName.isEmpty() )
        {
            QPixmap pixmap = QPixmap::grabWidget(this);

            const int width =save_width->value();
            const int heigth = save_heigth->value();

            if ( pixmap.scaled(width,heigth).save(fileName, "png" )){
                qDebug()<<"ok";
            }
            else
            {
                qDebug()<<"Uhmm";
            }
        }
    }
};

/* A class for retrieving topology graphs from
   adjacency matrices and for viewing these graphs */
class TopologyViewer: public QWidget {
    Q_OBJECT

  private:
  	  static const QString my_sign; // sign for log messages
  	
  	  TVWidget main_wdg; // part of TopologyViewer's functionality closely connected to drawing graphs
  	  QHBoxLayout *hor_layout; // layout-helper for placing 'main_wdg'
  	
  	  struct _for_NetCDF {
		  int v_file; // descriptor of data file in NetCDF format
		  int d_file; // descriptor of file with deviations (or of file to compare with) in NetCDF format
		  int matr_v,matr_d; // IDs of 3D matrices in NetCDF files
      } *ncdf_files; // to process NetCDF files

      struct _for_Text {
		  fpos_t matr_v_pos,matr_d_pos; // positions of 3D matrices in txt files
		  FILE *v_file; // descriptor of data file in txt format
		  FILE *d_file; // descriptor of file with deviations (or of file to compare with) in txt format
		  char flt_pt; // decimal point character (',' or '.' in floating-point numbers)
      } *txt_files; // to process TXT files

  	  int begin_message_length; // initial message length (obtained from file(s))
      int step_length; // step between two message lengths (obtained from file(s))

      /* topology retrieving */
      double shmem_eps; // consider all edges which interconnect one group of processors
      					// as placed on shared memory;
      					// 'shmem_eps' is the ratio of the maximum and minimum lengths among these edges;
      					// it's something close to 1.0
      double duplex_eps; // maximum acceptable ratio of lengths of two edges
      					 // connecting one pair of vertices; something close to 1.0

      /* graph building */
      unsigned char vals_for_edgs; // variant of what shall we use to compute lengths of the edges:
      							   // 0 - use simple average of values corresponding to different
      							   //     message lengths;
      							   // 1 - use median of values corresponding to different message lengths;
      							   // 2 - use exact values from user defined message length
      unsigned int usr_z_num; // user defined message length for the case when 'vals_for_edgs_var' is equal to 2
      double m_d_impact; // an importance of maximization of distances between unconnected vertices (0.0 .. 1.0);
      					 // if 'm_d_impact' is 0.0 then vertices are allowed to have the same coordinates;
      					 // increasing 'm_d_impact' means increasing distances between unconnected vertices
      					 // while increasing (!) the overall error in placement of the vertices
      unsigned char m_d_imp_tries; // number of tries to find optimum value for 'm_d_impact'

      /* misc */
      QString fname_for_tvwidget; // file name for title of TVWidget when it shows an "ideal" topology
      bool hosts_undefined; // 'true' if hosts' names are undefined

  private:
      Q_DISABLE_COPY(TopologyViewer)

      // the only constructor
      //
      // 'parent' - parent widget of this TopologyViewer
      // 'f_type' - type of input file(s)
      // 'was_error' - equals to 'true' if any errors occured
      //
      // Don't forget to call Init() after!
      TopologyViewer (QWidget *parent, const IData::Type f_type, bool &was_error);

      // initializes all members; must be called once after the constructor
      //
      // 'two_files' - 'true' if the file with deviations was loaded too
      // 'data_fname' - name of main data file
      // 'deviat_filename' - name of file with deviations of data, containing in 'data_fname'
      // 'hosts_fname' - file with hosts' names (that is names of graph's vertices)
      //
      // returns 'true' if there were no errors; in case of 'false' the use of TopologyViewer is denied!
      bool Init (const bool two_files, const QString &data_fname, const QString &deviat_filename,
      			 const QString &hosts_fname);

  public:
      // replaces calls to both the constructor and Init() (it was designed to ensure a call to Init());
      // returns 'NULL' if any errors occured;
      // returned pointer should be deleted using 'delete'
      static TopologyViewer* Create (QWidget *parent, const bool two_files, const IData::Type f_type,
      								 const QString &data_fname, const QString &deviat_filename,
      								 const QString &hosts_fname) {
          TopologyViewer *tp_v;
          bool was_error=false;

          try {
	      	  tp_v=new TopologyViewer(parent,f_type,was_error);
	      }
	      catch (const std::bad_alloc&) {
	      	  return NULL;
	      }
	      if (was_error || !tp_v->Init(two_files,data_fname,deviat_filename,hosts_fname))
		  {
		      delete tp_v;
		      return NULL;
		  }
		  return tp_v;
      }

      // destructor
      ~TopologyViewer ();

  private:
      Q_SLOT void Execute (void);


      //
      // returns 'false' if there was not enough memory
      bool RetrieveTopology (double *by_what_values);

      // fills 'matr' with values from input file
      //
      // 'vals_for_edgs' - determines the choice of message length
      // 'matr' - buffer for values; must have at least 'TVWidget::x_num*x_num' elements
      //
      // returns 'false' if there was not enough memory
      bool GetMatrixByValsForEdgsVar (double *matr);

      // compares retrieved topology with "ideal" topology loaded from file in DOT format
      void CompareTopologies (void);

  protected:
  	  // processes keyboard
      virtual void keyPressEvent (QKeyEvent*);

  Q_SIGNALS:
  	  // is connected to MainWindow::AddMessToLog()
  	  void SendMessToLog (const MainWindow::MsgType, const QString&, const QString&);
  	
  	  // is connected to MainWindow::ChangeTabTitle()
  	  void TitleChanged (QWidget*, const QString&);
  	
  	  // 'Esc' will close the tab with this viewer;
  	  // is connected to MainWindow::CloseTab()
      void CloseOnEscape (QWidget*);
};

/* 'Options' dialog for TopologyViewer */
class TopoViewerOpts: public QDialog {
	Q_OBJECT
	
  protected:
  	  virtual void keyPressEvent (QKeyEvent*);
  	
  public:
  	  TopoViewerOpts (QWidget *parent): QDialog(parent) {}
  	
      Q_SLOT void ShowMaxDistHelp (void);
};

