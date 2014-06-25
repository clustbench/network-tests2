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
#include "../core/renderer.h"
#include "render_opts.h"
#include <QPainter>
#include <QImage>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QTextEdit>
#include <QPushButton>
#include <QTextBlock>
#include <cmath>
#include "qexpandbox.h"
#include "mainwindow.h"

/* zooming options */
#define INC true
#define DEC false

class RotOXYButton;
class HostsBrowser;

class FullViewer: public QWidget {
	Q_OBJECT
	
	friend class RenderOpts;
	friend class RotOXYButton;
	friend class HostsBrowser;
  
  private:
	  /* FullViewer environment */
	  static const QString my_sign; // sign for log messages
	  
	  QString tab_name; // constant part of the name of this tab
	  
	  const int working_mode; // 0 - single file, 1 - data + deviations, 2 - compare two files
	  IData *v_file; // handles input data file
	  IData *d_file; // handles input file with deviations (or second data file)
	  
	  static const int IMAGE_OFFS=160; // x-position of the image
	  QImage *image; // result image descriptor
	  unsigned int img_w,img_h; // image width and height
	  unsigned int *pixels; // image pixels
	  QPainter *painter; // "easel"
	  QPushButton *draw_box_btn; // "Render!" button for calls to main render function
	  QPushButton *controls_btn; // shows hot keys and mouse usage
	  
	  QExpandBox *info_wdg; // container of info and options widgets
	  
	  QTextEdit *info; // file information
	  
	  HostsBrowser *hosts; // enumerated (from 1) hosts' names
	  unsigned int hosts_ind; // index of 'hosts' in 'info_wdg' for enabling/disabling 'hosts'
	  QPushButton *hosts_info; // informs that the text in 'hosts' is selectable and 
	  						   // selecting it means selecting "points"
	  
	  QTextEdit *dev_info; // device information
	  
	  /* main box renderer (based on OpenMP or OpenCL) */
	  Renderer *renderer;
	  
	  /* main box rendering (with transparency) */
	  /*const*/ double min_val1,max_val1; // minimum and maximum values in the first data file
	  /*const*/ double min_val2,max_val2; // minimum and maximum values in the second data file
	  unsigned short *clr_matrix; // according to 'working_mode' variable:  
	  							  // represents values, values+deviations or differences
	  int x_num,y_num,z_num; // sizes of color matrix's dimensions (that is number of points)
	  
	  RenderOpts *opts; // widget with options for renderers
	  unsigned int opts_ind; // index of 'opts' in 'info_wdg' for disabling 'opts' during "point selection" mode
	  
	  bool first_render; // allows to define the variables above only once
	  
	  bool volume_mode; // 'true' if "volume building" mode is on 
	  
	  /* mouse tracker */
	  bool mouse_pressed; // 'true' while mouse button is pressed
	  bool click_disabled; // 'true' if mouse was moved while its button is pressed
	  int x_move,y_move; // mouse movement while its button is pressed
	  
	  RotOXYButton *oxy_rot; // button for rotation in OXY
	  
	  /* point selection */
	  int sel_cube_x[8],sel_cube_y[8]; // corner points of selection cube (selection rectangle in 3D): 
	  								   // [0] = left bottom near, [1] = right bottom near, 
	  								   // [2] = left top near, [3] = right top near, 
	  								   // [4] = left bottom far, [5] = right bottom far, 
	  								   // [6] = left top far, [7] = right top far
	  bool sel_cube_vis[8]; // visibility of selection cube's points
	  
	  bool pt_selection; // 'true' if "point selection" mode is on
	  bool pt_selected; // 'true' if some "point" is selected (by mouse click)
	  bool hst_selected; // 'true' if host name or host index is selected in 'hosts' widget
	  
	  QTextEdit *pt_info; // information about selected "point"

  private:
	  Q_DISABLE_COPY(FullViewer)
	  
	  // the only constructor
	  //
	  // 'parent' - the widget that creates this viewer (it is important for signal connecting)
	  // 'mode' defines 'working mode'
	  // 'was_error' - returns whether there were errors in the constructor
	  //
	  // don't forget to call Init() function!
	  FullViewer (QWidget *parent, const int mode, bool &was_error);
	  
	  // initializes all members; must be called once after the constructor
	  //
	  // 'data_filename' - the name of file with exact values
	  // 'deviat_filename' - the name of file with values' deviations
	  // 'hosts_filename' - the name of file with hosts' names in case of NetCDF data files; empty string otherwise
	  // 'f_type' - type of input file(s)
	  // 'tab_nm' - name of this tab
	  //
	  // returns 'true' if there were no errors; in case of 'false' you cannot use FullViewer! */
	  bool Init (const QString &data_filename, const QString &deviat_filename, const QString &hosts_filename, 
	  			 const IData::Type f_type, const QString &tab_nm);

  public:
	  // replaces calls to both the constructor and Init() (it was designed to ensure the call to Init()); 
	  // returns 'NULL' if any errors occured; 
	  // returned pointer should be deleted using 'delete'
	  static FullViewer* Create (QWidget *parent, const IData::Type f_type, const int mode, 
	  							 const QString &data_filename, const QString &deviat_filename, 
	  							 const QString &hosts_filename, const QString &tab_nm) {
		  FullViewer *fullv;
		  bool was_error;
		  
		  try {
			  fullv=new FullViewer(parent,mode,was_error);
		  }
		  catch (const std::bad_alloc&) {
		  	  return NULL;
		  }
		  if (was_error || !fullv->Init(data_filename,deviat_filename,hosts_filename,f_type,tab_nm))
		  {
		  	  delete fullv;
		  	  return NULL;
		  }
		  return fullv;
	  }
	  
	  // destructor
	  ~FullViewer ();

  protected:
	  // redraws window
	  virtual void paintEvent (QPaintEvent*) {
	  	  painter->begin(this);
	  	  painter->setRenderHint(QPainter::Antialiasing,false);
	  	  painter->drawImage(IMAGE_OFFS,1,*image);
	  	  if (first_render)
	  	  {
	  	  	  QFont f(painter->font());
	  	  	  f.setPixelSize(18);
	  	  	  painter->setFont(f);
	  	  }
	  	  else
	  	  {
	  	  	  painter->setRenderHint(QPainter::Antialiasing,true);
	  	  	  
	  	  	  /* draw axes in right upper corner of main image */
	  	  	  static float axs[6];
	  	  	  static const QPen red_pen(Qt::red),blue_pen(Qt::blue),yellow_pen(Qt::yellow);
	  	  	  const int origin_x=IMAGE_OFFS+static_cast<int>(img_w)-70,origin_y=60; // origin of all axes
	  	  	  static const float scale=40.0f; // length of all axes
	  	  	  int to_x,to_y;
	  	  	  renderer->GetAxesForDrawing(axs);
	  	  	  // Ox
	  	  	  painter->setPen(red_pen);
	  	  	  to_x=static_cast<int>(floorf(static_cast<float>(origin_x)+axs[0]*scale+0.5f));
	  	  	  to_y=static_cast<int>(floorf(static_cast<float>(origin_y)+axs[1]*scale+0.5f));
	  	  	  painter->drawLine(origin_x,origin_y,to_x,to_y);
	  	  	  painter->drawText(to_x+3,to_y+3,tr("Snd")); // "senders"
			  // Oy
			  painter->setPen(blue_pen);
			  to_x=static_cast<int>(floorf(static_cast<float>(origin_x)+axs[2]*scale+0.5f));
			  to_y=static_cast<int>(floorf(static_cast<float>(origin_y)+axs[3]*scale+0.5f));
			  painter->drawLine(origin_x,origin_y,to_x,to_y);
			  painter->drawText(to_x+3,to_y+3,tr("Rcv")); // "receivers"
			  // Oz
			  painter->setPen(yellow_pen);
			  to_x=static_cast<int>(floorf(static_cast<float>(origin_x)+axs[4]*scale+0.5f));
			  to_y=static_cast<int>(floorf(static_cast<float>(origin_y)+axs[5]*scale+0.5f));
			  painter->drawLine(origin_x,origin_y,to_x,to_y);
			  painter->drawText(to_x+3,to_y+3,tr("ML")); // "message lengths"
			  
			  /* area for OXY-rotation button */
			  static const QPen OXY_pen(QColor(200,200,200,127));
			  static const QString oxy("OXY");
			  painter->setPen(OXY_pen);
			  painter->drawText(IMAGE_OFFS+18,img_h-24u,oxy);
			  painter->drawArc(IMAGE_OFFS+9,img_h-54u,48,48,0,5760);
			  painter->drawArc(IMAGE_OFFS+5,img_h-58u,56,56,0,5760);
			  
			  if (pt_selection && pt_selected)
			  {
				  painter->setClipRect(IMAGE_OFFS,1,img_w,img_h); // make selection cube not exceeding the borders
				  painter->setPen(blue_pen);
				  if (sel_cube_vis[0])
				  {
					  if (sel_cube_vis[1])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[0],1+sel_cube_y[0],IMAGE_OFFS+sel_cube_x[1],1+sel_cube_y[1]);
					  if (sel_cube_vis[2])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[0],1+sel_cube_y[0],IMAGE_OFFS+sel_cube_x[2],1+sel_cube_y[2]);
					  if (sel_cube_vis[4])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[0],1+sel_cube_y[0],IMAGE_OFFS+sel_cube_x[4],1+sel_cube_y[4]);
				  }
				  if (sel_cube_vis[6])
				  {
					  if (sel_cube_vis[7])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[6],1+sel_cube_y[6],IMAGE_OFFS+sel_cube_x[7],1+sel_cube_y[7]);
					  if (sel_cube_vis[4])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[6],1+sel_cube_y[6],IMAGE_OFFS+sel_cube_x[4],1+sel_cube_y[4]);
					  if (sel_cube_vis[2])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[6],1+sel_cube_y[6],IMAGE_OFFS+sel_cube_x[2],1+sel_cube_y[2]);
				  }
				  if (sel_cube_vis[3])
				  {
					  if (sel_cube_vis[2])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[3],1+sel_cube_y[3],IMAGE_OFFS+sel_cube_x[2],1+sel_cube_y[2]);
					  if (sel_cube_vis[1])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[3],1+sel_cube_y[3],IMAGE_OFFS+sel_cube_x[1],1+sel_cube_y[1]);
					  if (sel_cube_vis[7])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[3],1+sel_cube_y[3],IMAGE_OFFS+sel_cube_x[7],1+sel_cube_y[7]);
				  }
				  if (sel_cube_vis[5])
				  {
					  if (sel_cube_vis[4])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[5],1+sel_cube_y[5],IMAGE_OFFS+sel_cube_x[4],1+sel_cube_y[4]);
					  if (sel_cube_vis[7])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[5],1+sel_cube_y[5],IMAGE_OFFS+sel_cube_x[7],1+sel_cube_y[7]);
					  if (sel_cube_vis[1])
					  painter->drawLine(IMAGE_OFFS+sel_cube_x[5],1+sel_cube_y[5],IMAGE_OFFS+sel_cube_x[1],1+sel_cube_y[1]);
				  }
	      	  }
		  }
		  painter->end();
	  }
	  
	  // processes keyboard
	  virtual void keyPressEvent (QKeyEvent *key_event) {
		  const int key=key_event->key();
		  
		  if (key==Qt::Key_Escape) // 'Esc' pressed -> close this tab
		  {
		  	  emit CloseOnEscape(this);
		  	  return;
		  }
		  if (first_render) return;
		  switch (key)
		  {
			case Qt::Key_A: // shift "camera" left
				renderer->ShiftCamera(-20.0f,0.0f,0.0f);
				RenderBox();
				break;
			case Qt::Key_D: // shift "camera" right
				renderer->ShiftCamera(20.0f,0.0f,0.0f);
				RenderBox();
				break;
			case Qt::Key_W: // shift "camera" up
				renderer->ShiftCamera(0.0f,-20.0f,0.0f);
				RenderBox();
				break;
			case Qt::Key_S: // shift "camera" down
				renderer->ShiftCamera(0.0f,20.0f,0.0f);
				RenderBox();
				break;
			case Qt::Key_Z: // shift "camera" backward (the box moves farther from observer)
				renderer->ShiftCamera(0.0f,0.0f,-20.0f);
				RenderBox();
				break;
			case Qt::Key_X: // shift "camera" forward (the box moves nearer to observer)
				renderer->ShiftCamera(0.0f,0.0f,20.0f);
				RenderBox();
				break;
			case Qt::Key_Plus: // increase box size
				if (renderer->Zoom(INC)) RenderBox();
				break;
			case Qt::Key_Minus: // decrease box size
				if (renderer->Zoom(DEC)) RenderBox();
				break;
		  }
	  }
	  
	  // processes mouse wheel
	  virtual void wheelEvent (QWheelEvent *wheel_event) {
		  if (!first_render && renderer->Zoom(wheel_event->delta()>0)) RenderBox();
	  }
	  
	  // processes left mouse button presses: initializes mouse position to rotate the bounding box (in future)
	  virtual void mousePressEvent (QMouseEvent *mouse_event) {
		  if (!first_render && (mouse_event->button()==Qt::LeftButton) && 
			  (mouse_event->x()>=IMAGE_OFFS) && (mouse_event->x()<=IMAGE_OFFS+static_cast<int>(img_w)) && 
			  (mouse_event->y()>0) && (mouse_event->y()<2+static_cast<int>(img_h)))
		  {
			  mouse_pressed=true;
			  click_disabled=false;
			  x_move=mouse_event->x();
			  y_move=mouse_event->y();
		  }
	  }
	  
	  // processes left mouse button releases
      virtual void mouseReleaseEvent (QMouseEvent *mouse_event) {
		  if (!first_render && (mouse_event->button()==Qt::LeftButton))
		  {
			  const int m_x=mouse_event->x()-IMAGE_OFFS,m_y=mouse_event->y()-1;
			  
			  mouse_pressed=false;
			  if ((m_x<0) || (m_x>=static_cast<int>(img_w)) || (m_y<0) || (m_y>=static_cast<int>(img_h)))
			  {
				  click_disabled=false;
				  return;
			  }
			  if (click_disabled) return;
			  if (pt_selection)
			  {
				  Coords pos;
				  
				  pt_selected=renderer->SelectPoints_last_phase(m_x,m_y,pos,clr_matrix);
				  if (pt_selected)
					  PointInfo(pos);
				  else
				  {
					  if (QMessageBox::warning(this,tr("\"Point selection\""),tr("Leave \"point selection\" mode?"),
								   			   QMessageBox::Yes | QMessageBox::No)==QMessageBox::Yes)
					  {
						  double new_min,new_max;
						  
						  pt_selection=false;
						  hst_selected=false;
						  if (pt_info!=NULL) pt_info->hide();
						  renderer->ToggleVolumeMode(volume_mode);
						  opts->GetClrStretchingMinMax(new_min,new_max); // to restore color stretching
						  ReFillMatrix(new_min,new_max,true);
						  renderer->TurnOffPtSelection();
						  info_wdg->setItemEnabled(opts_ind,true);
					  }
				  }
				  RenderBox(); // update() is there
			  }
			  else
			  {
				  Coords *pos=NULL;
				  int num=0;
				  
				  renderer->SelectPoints_by_click(m_x,m_y,hst_selected? clr_matrix : NULL,pos,num);
				  if (num!=0)
				  {
					  pt_selection=true;
					  info_wdg->setItemEnabled(opts_ind,false); // render options are unavailable in 
					  											// "point selection" mode
					  LeaveChosenInMatrix(pos,num);
					  if (!hst_selected)
					  {
						  volume_mode=renderer->IsVolumeMode();
						  renderer->ToggleVolumeMode(true);
					  }
					  if (num==1)
					  {
						  // immediately choose this only "point"
						  pt_selected=renderer->SelectPoints_last_phase(m_x,m_y,pos[0],clr_matrix);
						  PointInfo(pos[0]);
					  }
					  renderer->FreePoints(pos);
					  RenderBox();
				  }
				  else
					  if (pos!=NULL) renderer->FreePoints(pos);
			  }
		  }
	  }
	  
	  // processes mouse movement: gets mouse movement vector to compute rotation angles
	  virtual void mouseMoveEvent (QMouseEvent *mouse_event) {
		  if (mouse_pressed && !first_render)
		  {
			  const int new_x=mouse_event->x(),new_y=mouse_event->y();
			  if ((new_x<IMAGE_OFFS) || (new_x>IMAGE_OFFS+(int)img_w) || (new_y<1) || (new_y>1+(int)img_h) || 
			  	  ((new_x<x_move+3) && (new_x+3>x_move) && (new_y<y_move+3) && (new_y+3>y_move))) return;
			  
			  click_disabled=true;
			  
			  renderer->Rotate(3.14159274101257f*static_cast<float>(new_x-x_move)/static_cast<float>(img_w),
							   3.14159274101257f*static_cast<float>(new_y-y_move)/static_cast<float>(img_h));
			  
			  // move vector origin to the new point
			  x_move=new_x;
			  y_move=new_y;
			  
			  RenderBox();
		  }
	  }

  Q_SIGNALS:
	  // is connected to MainWindow::AddMessToLog()
	  void SendMessToLog (const MainWindow::MsgType, const QString&, const QString&);
	  
	  // is connected to MainWindow::ChangeTabTitle()
	  void TitleChanged (QWidget*, const QString&);
	  
	  // 'Esc' will close the tab with this viewer; 
	  // is connected to MainWindow::CloseTab()
	  void CloseOnEscape (QWidget*);

  public Q_SLOTS:
	  // makes 'renderer' do rotation in OXY plane 
	  // (this is the response to 'oxy_rot' button signal)
	  void OXY_Rotation (const float cosa, const bool positive) {
		  renderer->RotateOXY(cosa,positive);
		  RenderBox();
	  }

  private:
	  // reads all data from input file(s) and fills color matrix
	  void FillMatrix ();
	  
	  // tries to decrease matrix size when there is not enough memory
	  void CompressMatrix () {
		  // TODO: compression (don't forget about x|y|z_size and point_size)!!
		  emit CloseOnEscape(this); // temporary!!
	  }
	  
	  // reads all data from input file(s) and fills color matrix according to 'new_min' and 'new_max'; 
	  // if 'show_msg' is 'true' a window with a message about refilling will be shown
	  void ReFillMatrix (const double new_min, const double new_max, const bool show_msg=false);
	  
	  // sets all colors, which correspond to values that are not in [new_min_v,new_max_v] and
	  // deviations that are not in [new_min_d,new_max_d], to 0
	  void AdjustMatrix (const double new_min_v, const double new_max_v, 
	  					 const double new_min_d, const double new_max_d);
	  
	  // leaves in color matrix ony those cells which are in the array 'pos';
	  // other cells will become zeroes
	  void LeaveChosenInMatrix (const Coords *const pos, const int num);
	  
	  // leaves in color matrix ony those cells which have their row or column 
	  // indices equal to 'proc_ind'; other cells will become zeroes
	  void LeaveChosenInMatrix (const unsigned int proc_ind);
	  
	  // leaves in color matrix ony those cells which have their row or column 
	  // indices equal to indices in 'sorted_pts'; other cells will become zeroes; 
	  // 'sorted_pts' must be sorted in ascending order!
	  // 'pt_num' is a size of the array 'sorted_pts' (must be greater than 1!)
	  void LeaveChosenInMatrix (const unsigned int *const sorted_pts, const unsigned int pt_num);
	  
	  // shows widget with information about selected "point" defined by coordinates 'pos'
	  void PointInfo (const Coords &pos);

  private Q_SLOTS:
	  // main render function
	  void RenderBox ();
	  
	  // shows hot keys and mouse usage
	  void ShowControls (void) {
		  QMessageBox::information(this,tr("Controls"),
					   tr("<div><span style=\"color:darkred\">'W', 'A', 'S', 'D'</span> - "
						  "move \"camera\" up/left/down/right</div>"
						  "<div><span style=\"color:darkred\">'Z', 'X'</span> - move \"camera\" to/from obserever</div>"
						  "<div><span style=\"color:darkred\">'+', '-', 'Scroll'</span> - zoom in/out</div>"
						  "<div><span style=\"color:darkred\">Press left mouse key and move mouse</span> - rotation</div>"
						  "<div><span style=\"color:darkred\">'Esc'</span> - close this tab</div>"));
	  }
	  
	  // shows information that the text in 'hosts' is selectable and selecting it means selecting of "points"
	  void ShowHostsInfo (void) {
		  QMessageBox::information(this,tr("Hosts: description"),tr("<b>1.</b> Double clicking on a <i><b>number</b></i> N selects<br>"
										   "all \"points\" with their <i>sender</i> or <i>receiver</i><br>"
										   "coordinates equal to (N-1).<br>"
										   "<b>2.</b> Double clicking on <i><b>host' name</b></i> selects all<br>"
										   "<i><b>numbers</b></i> corresponding to equal names"));
	  }

  public:
	  // changes representation of "points"
	  void ChangePtRepr (const PtRepr);
	  
	  // sets new value for 'depth_constraint'
	  void SetDepthConstraint (const unsigned int new_d_c) {
		  renderer->SetDepthConstraint(new_d_c);
		  if (!first_render) RenderBox();
	  }
};

class RotOXYButton: public QPushButton {
	Q_OBJECT

  private:
	  const float c_x,c_y; // centre of the button's rotation circle
	  int m_x,m_y; // relative mouse coordinates
	  float ms_x_init,ms_y_init; // vector from (c_x,c_y) to mouse click position
	  float ms_ang; // ==asinf(ms_y_init)

  public:
	  RotOXYButton (FullViewer *par, const int par_img_h): 
	    QPushButton(par), c_x(FullViewer::IMAGE_OFFS+8+25), c_y(par_img_h-55+25) {
		  this->setFixedSize(16,14);
		  this->move(FullViewer::IMAGE_OFFS,par_img_h-55+18);
		  connect(this,SIGNAL(NeedRender(float,bool)),par,SLOT(OXY_Rotation(const float,const bool)));
	  }

  Q_SIGNALS:
	  // notifies FullViewer about made rotation
	  void NeedRender (float, bool);

  protected:
	  // processes left mouse button presses: initializes mouse position to rotate the bounding box (in future)
	  void mousePressEvent (QMouseEvent *mouse_event) {
		  if (mouse_event->button()==Qt::LeftButton)
		  {
			  grabMouse();
			  this->setDown(true); // push the button
			  /* save relative mouse coordinates to move the button precisely */
			  m_x=mouse_event->x();
			  m_y=mouse_event->y();
			  /* from centre of circle to point of mouse click */
			  ms_x_init=static_cast<float>(x()+mouse_event->x())-c_x;
			  ms_y_init=static_cast<float>(y()+mouse_event->y())-c_y;
			  const float len=sqrtf(ms_x_init*ms_x_init+ms_y_init*ms_y_init);
			  ms_x_init/=len;
			  ms_y_init/=len;
			  ms_ang=asinf(ms_y_init);
		  }
	  }
	  
	  // processes left mouse button releases: gets mouse movement vector to compute rotation angles
	  void mouseReleaseEvent (QMouseEvent *mouse_event) {
		  if (mouse_event->button()==Qt::LeftButton)
		  {
			  releaseMouse();
			  this->setDown(false); // release the button
		  }
	  }
	  
	  // processes mouse movement
	  void mouseMoveEvent (QMouseEvent *mouse_event) {
		  //move(x()+mouse_event->x()-ms_x_when_clicked,y()+mouse_event->y()-ms_y_when_clicked); /* button moves with mouse!!:) */
		  
		  /* The idea:
			 when you click this button and hold mouse button pressed, centre of the circle mentioned above 
			 becomes "centre of the world" and all mouse coordinates are projected onto the circle. 
			 And centre of the button will always be on the line joining mouse position with "centre of the world". */
		  
		  /* from centre of circle to point of mouse click */
		  float ms_x=static_cast<float>(x()+mouse_event->x())-c_x;
		  float ms_y=static_cast<float>(y()+mouse_event->y())-c_y;
		  const float len=sqrtf(ms_x*ms_x+ms_y*ms_y);
		  
		  if (len<0.00001f) return; // too close to centre of the circle!
		  
		  ms_x/=len;
		  ms_y/=len;
		  
		  /* rotate bounding box */
		  const float m_ang=asinf(ms_y);
		  emit NeedRender(1.0f-((ms_x-ms_x_init)*(ms_x-ms_x_init)+(ms_y-ms_y_init)*(ms_y-ms_y_init))*0.5f,
		  				  (ms_x>=0.0f) ^ (m_ang<ms_ang));
		  
		  /* move button itself */
		  move(static_cast<int>(floor(c_x+25.0f*ms_x+0.5f))-m_x,static_cast<int>(floor(c_y+25.0f*ms_y+0.5f))-m_y);
		  
		  ms_ang=m_ang;
		  /* move "click" vector */
		  ms_x_init=ms_x;
		  ms_y_init=ms_y;
	  }
};

// Had to add this class so as "connect(QTextEdit,SIGNAL(selectionChanged()),FullViewer,SLOT(...))" 
// is suboptimal because of very frequent signal sending
class HostsBrowser: public QTextEdit {
	Q_OBJECT
      
  public:
	  HostsBrowser (QWidget *parent): QTextEdit(parent) {}
      
  protected:
	  // processes mouse double ckicks and tells FullViewer 
	  // about entering "point selection" mode;
	  // 'FullViewer::first_render' MUST BE 'FALSE'!
	  void mouseDoubleClickEvent (QMouseEvent *mouse_event) {
		  if (mouse_event->button()!=Qt::LeftButton) return;
		  
		  FullViewer *parent=static_cast<FullViewer*>(parentWidget());
		  
		  if (parent->first_render) return;
		  
		  // let QTextEdit handle the event and do the selection; 
		  // although 'NoTextInteraction' mode is set, the widget 
		  // continues to receive mouse events
		  setTextInteractionFlags(Qt::TextSelectableByMouse);
		  QTextEdit::mouseDoubleClickEvent(mouse_event);
		  setTextInteractionFlags(Qt::NoTextInteraction);
		  
		  QTextCursor curs=textCursor();
		  
		  if (!curs.hasSelection()) return;
		  
		  QTextBlock blk=curs.block();
		  
		  if (blk.contains(curs.anchor()))
		  {	  
			  QString what=curs.selectedText();
			  
			  if (what.indexOf(')')>=0) return; // neither a number nor host name
			  
			  QString where=blk.text();
			  double new_min,new_max;
			  
			  if (parent->pt_selection)
			  {
				  parent->pt_selection=false;
				  parent->pt_selected=false;
				  if (parent->pt_info!=NULL) parent->pt_info->hide();
				  parent->opts->GetClrStretchingMinMax(new_min,new_max);
				  parent->ReFillMatrix(new_min,new_max,true);
				  parent->renderer->TurnOffPtSelection();
			  }
			  else
			  {
				  if (parent->hst_selected)
				  {
					  // in this case initial values in color matrix was partly erased
					  parent->opts->GetClrStretchingMinMax(new_min,new_max);
					  parent->ReFillMatrix(new_min,new_max,true);
				  }
				  else
				  {
					  parent->volume_mode=parent->renderer->IsVolumeMode();
					  parent->renderer->ToggleVolumeMode(true);
				  }
			  }
			  parent->info_wdg->setItemEnabled(parent->opts_ind,false);
			  
			  parent->hst_selected=true;
			  if (where.indexOf(what)==0)
			  {
				  // nearly sure that a number was selected
				  
				  bool ok;
				  const unsigned int proc_ind=what.toUInt(&ok)-1u;
				  
				  if (ok && (proc_ind<static_cast<unsigned int>(parent->x_num)))
					  // a number was selected
					  parent->LeaveChosenInMatrix(proc_ind);
				  else
				  {
					  QMessageBox::critical(parent,tr("Hosts: error"),
					  						tr("Invalid number: '")+QString::number(proc_ind)+"'!");
					  parent->hst_selected=false;
					  return;
				  }
			  }
			  else
			  {
				  // assume that host name was selected
				  
				  // select the whole host name
				  curs.movePosition(QTextCursor::StartOfLine,QTextCursor::MoveAnchor);
				  curs.movePosition(QTextCursor::WordRight,QTextCursor::MoveAnchor,2);
				  curs.movePosition(QTextCursor::EndOfLine,QTextCursor::KeepAnchor);
				  setTextCursor(curs);
				  
				  unsigned int pt_num=0u,pt_mem_num=5u;
				  unsigned int *pts=static_cast<unsigned int*>(malloc(pt_mem_num*sizeof(int)));
				  
				  where.remove(0,where.indexOf(' ')+1); // leave in 'where' only host name
				  curs.movePosition(QTextCursor::Start); // to the top of text edit
				  for (blk=curs.block(); blk.isValid(); blk=blk.next())
				  {
					  what=blk.text();
					  if (what.contains(where))
					  {
						  if (pt_num==pt_mem_num)
						  {
							  pt_mem_num<<=1u; // double memory consumption
							  pts=static_cast<unsigned int*>(realloc(pts,pt_mem_num*sizeof(int)));
						  }
						  pts[pt_num]=(what.left(what.indexOf(')'))).toUInt()-1u;
						  ++pt_num;
					  }
				  }
				  if (pt_num==1u) parent->LeaveChosenInMatrix(pts[0]);
				  else parent->LeaveChosenInMatrix(pts,pt_num);
				  free(pts);
			  }
			  parent->RenderBox();
		  }
		  else
			  QMessageBox::critical(parent,tr("Hosts: error"),
			  						tr("More than one line is selected.<br>Cannot proceed"));
	  }
};

