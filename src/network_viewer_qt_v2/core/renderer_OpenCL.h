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

#include "opencl_defs.h" // must be generated automatically
#include "renderer.h"
#include <cmath>
#include <QObject> // for class QString and function tr()

/* OpenCL-based renderer */
class RendererOCL: public Renderer {
  private:
	  cl_mem gpu_pixels; // image pixels on GPU
	  unsigned int img_w,img_h; // image width and height (will be received from FullViewer)
	  
	  cl_float16 data; /* data of bounding box of all "points": 
	  					  .s012345678 - coordinate axes: .s012=Ox={1,0,0}, .s345=Oy={0,-1,0}, .s678=Oz={0,0,1};
	  					  .s9ab - "camera" position minus centre of bounding box;
	  					  .sc - distance between two "points"
	  					  .sd - square of "point" radius */
	  
	  unsigned int depth_constraint; // maximum number of "points" which can intersect with one ray
	  
	  /* main box rendering (with transparency) */
	  cl_mem matrix; // color matrix
	  cl_int3 num; // color matrix dimensions (number of points)
	  
	  /* OpenCL renderer environment */
	  cl_context contxt; // device context
	  cl_program program; // program written in OpenCL with rendering kernels
	  cl_kernel kernel; // rendering kernel function
	  cl_command_queue comm_queue; // queue for kernels execution
	  
	  PtRepr pt_repr; // representation of "points"
	  
	  cl_uchar4 clr_minmax; // all color constraints
	  
	  bool volume_mode; // 'true' in "volume building" mode
	  
	  /* "point selection" */
	  bool pt_selection_mode;
	  cl_int3 sel_ind; // coordinates of selected "point" (are valid only during "point selection" mode!)
	  int *const fv_sel_cube_x,*const fv_sel_cube_y; // coordinates of selection cube
	  bool *const fv_sel_cube_vis; // visibility of selection cube's points
	  
	  QString err_str; // contains error messages

  public:
	  // constructor
	  RendererOCL (int *const fv_sl_cube_x, int *const fv_sl_cube_y, bool *const fv_sl_cube_vis);
	  
	  // must be called once after constructor
	  void Init (const unsigned int image_w, const unsigned int image_h, 
	  			 const int num_x, const int num_y, const int num_z);
	  
	  // destructor
	  virtual ~RendererOCL ();
	  
	  // returns error string
	  const QString& GetError (void) const { return err_str; }
	  
	  // returns the number of compute units on the device
	  unsigned int NumberCUs (void) const {
		  cl_device_id dev_id;
		  cl_uint max_CUs;
		  
		  /* note that only one device is used! */
		  clGetContextInfo(contxt,CL_CONTEXT_DEVICES,sizeof(cl_device_id),&dev_id,NULL);
		  clGetDeviceInfo(dev_id,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&max_CUs,NULL);
		  return max_CUs;
	  }
	  
	  // gets 'clr_matrix' variable from FullViewer
	  void SetClrMatrix (unsigned short *clr_matrix) {
		  if (matrix!=NULL)
			  clEnqueueWriteBuffer(comm_queue,matrix,CL_FALSE,0,num.x*num.y*num.z*sizeof(short),clr_matrix,0,NULL,NULL);
		  else
		  {
		  	  matrix=clCreateBuffer(contxt,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,num.x*num.y*num.z*sizeof(short),
		  	  						clr_matrix,NULL);
		  	  clSetKernelArg(kernel,3,sizeof(cl_mem),&matrix);
		  }
	  }
	  
	  // shifts "camera" origin by vector {sh_x,sh_y,sh_z}
	  void ShiftCamera (const float sh_x, const float sh_y, const float sh_z) {
		  data.s9+=sh_x;
		  data.sa+=sh_y;
		  data.sb+=sh_z;
		  clSetKernelArg(kernel,0,sizeof(cl_float16),&data);
		  if (pt_selection_mode)
			  BuildSelectionCube(data.s9*data.s0+data.sa*data.s1+data.sb*data.s2+((float)num.x)*data.sc*0.5f,
								 data.s9*data.s3+data.sa*data.s4+data.sb*data.s5+((float)num.y)*data.sc*0.5f,
								 data.s9*data.s6+data.sa*data.s7+data.sb*data.s8+((float)num.z)*data.sc*0.5f);
	  }
	  
	  // increases/decreases bounding box size (according to 'inc') by increasing/decreasing 'point_step'
	  bool Zoom (const bool inc) {
		  const float new_point_step=data.sc+(inc? 0.5f : (-0.5f));
		  if (new_point_step<=0.0f) return false;
		  
		  const float koeff=new_point_step/data.sc;
		  
		  data.sd*=(koeff*koeff);
		  data.sc=new_point_step;
		  
		  clSetKernelArg(kernel,0,sizeof(cl_float16),&data);
		  
		  if (pt_selection_mode)
			  BuildSelectionCube(data.s9*data.s0+data.sa*data.s1+data.sb*data.s2+((float)num.x)*data.sc*0.5f,
								 data.s9*data.s3+data.sa*data.s4+data.sb*data.s5+((float)num.y)*data.sc*0.5f,
								 data.s9*data.s6+data.sa*data.s7+data.sb*data.s8+((float)num.z)*data.sc*0.5f);
		  return true;
	  }
	  
	  // rotates bounding box by 'w_angle' in OXZ and 'h_angle' in OYZ
	  void Rotate (const float w_angle, const float h_angle) {
		  // rotation in OXZ
		  float sina=sinf(w_angle),cosa=cosf(w_angle);
		  float tmp=data.s0;
		  data.s0=tmp*cosa-data.s2*sina;
		  data.s2=tmp*sina+data.s2*cosa;
		  tmp=data.s3;
		  data.s3=tmp*cosa-data.s5*sina;
		  data.s5=tmp*sina+data.s5*cosa;
		  tmp=data.s6;
		  data.s6=tmp*cosa-data.s8*sina;
		  data.s8=tmp*sina+data.s8*cosa;
		  
		  // rotation in OYZ
		  sina=sinf(h_angle);
		  cosa=cosf(h_angle);
		  tmp=data.s1;
		  data.s1=tmp*cosa-data.s2*sina;
		  data.s2=tmp*sina+data.s2*cosa;
		  tmp=data.s4;
		  data.s4=tmp*cosa-data.s5*sina;
		  data.s5=tmp*sina+data.s5*cosa;
		  tmp=data.s7;
		  data.s7=tmp*cosa-data.s8*sina;
		  data.s8=tmp*sina+data.s8*cosa;
		  
		  clSetKernelArg(kernel,0,sizeof(cl_float16),&data);
		  
		  if (pt_selection_mode)
			  BuildSelectionCube(data.s9*data.s0+data.sa*data.s1+data.sb*data.s2+((float)num.x)*data.sc*0.5f,
								 data.s9*data.s3+data.sa*data.s4+data.sb*data.s5+((float)num.y)*data.sc*0.5f,
								 data.s9*data.s6+data.sa*data.s7+data.sb*data.s8+((float)num.z)*data.sc*0.5f);
	  }
	  
	  // rotates bounding box by arccos('cosa') in OXY
	  void RotateOXY (const float cosa, const bool positive) {
		  const float sina=sqrtf(1.0f-cosa*cosa);
		  float tmp=data.s0;
		  
		  if (positive)
		  {
			  data.s0=tmp*cosa-data.s1*sina;
			  data.s1=tmp*sina+data.s1*cosa;
			  tmp=data.s3;
			  data.s3=tmp*cosa-data.s4*sina;
			  data.s4=tmp*sina+data.s4*cosa;
			  tmp=data.s6;
			  data.s6=tmp*cosa-data.s7*sina;
			  data.s7=tmp*sina+data.s7*cosa;
		  }
		  else
		  {
			  data.s0=tmp*cosa+data.s1*sina;
			  data.s1=data.s1*cosa-tmp*sina;
			  tmp=data.s3;
			  data.s3=tmp*cosa+data.s4*sina;
			  data.s4=data.s4*cosa-tmp*sina;
			  tmp=data.s6;
			  data.s6=tmp*cosa+data.s7*sina;
			  data.s7=data.s7*cosa-tmp*sina;
		  }
		  
		  clSetKernelArg(kernel,0,sizeof(cl_float16),&data);
		  
		  if (pt_selection_mode)
			  BuildSelectionCube(data.s9*data.s0+data.sa*data.s1+data.sb*data.s2+((float)num.x)*data.sc*0.5f,
								 data.s9*data.s3+data.sa*data.s4+data.sb*data.s5+((float)num.y)*data.sc*0.5f,
								 data.s9*data.s6+data.sa*data.s7+data.sb*data.s8+((float)num.z)*data.sc*0.5f);
	  }
	  
	  // writes projection on the screen of all 3 axes to 'axes';
	  // the projection is computed by simply throwing Z-coordinates away;
	  // 'axes' must have at least 6 elements:
	  //    [0],[1] - Ox;
	  //    [2],[3] - Oy;
	  //    [4],[5] - Oz
	  void GetAxesForDrawing (float *axes) const {
		  axes[0]=data.s0;
		  axes[1]=data.s1;
		  axes[2]=data.s3;
		  axes[3]=data.s4;
		  axes[4]=data.s6;
		  axes[5]=data.s7;
	  }
	  
	  /// main render function
	  void RenderBox (unsigned int *pixels) const;
	  
	  // chooses variants of RenderBox according to 'pt_rpr'
	  PtRepr ChangeKernel (const PtRepr pt_rpr);
	  
	  // sets new value for 'depth_constraint'
	  void SetDepthConstraint (const unsigned int new_d_c) {
		  depth_constraint=new_d_c;
		  if (!volume_mode) // do NOT set depth constraint in "volume building" mode!
			  clSetKernelArg(kernel,2,sizeof(int),&depth_constraint);
	  }
	  
	  // returns 'true' in "volume building" mode
	  bool IsVolumeMode (void) const { return volume_mode; }
	  
	  // turns on/off "volume building" mode;
	  bool ToggleVolumeMode (const bool on);
	  
	  // removes "points" with green color not in [min_green,max_green] and red color not in [min_red,max_red]
	  // must be called ONLY after ToggleVolumeMode(true)!!
	  bool BuildVolume (const unsigned char min_green, const unsigned char max_green, 
						const unsigned char min_red, const unsigned char max_red);
	  
	  // fills array 'pos' with coordinates of visible "points" in 'clr_matrix' 
	  // which are intersected with the ray that goes through the point (x,y);
	  // 'points_num' is the number of such "points" and the size of the array too
	  void SelectPoints_by_click (const int x, const int y, const unsigned short *const clr_matrix, 
	  							  Coords* &pos, int &points_num) const;
	  
	  // frees array of points obtained by SelectPoints_by_click()
	  void FreePoints (Coords *pts) const { free(pts); }
	  
	  // in "point selection" mode gives coordinates (to 'pos') of chosen point (the nearest to observer!);
	  // 'clr_matrix' is color matrix obtained from FullViewer
	  // 'sel_cube_x' will get x-coordinates of all 8 corners of hit "point", 
	  // 'sel_cube_y' will get y-coordinates of all 8 corners of hit "point",
	  // 'sel_cube_vis' will get visibility of all 8 corners of hit "point";
	  // returns 'true' if some "point" was hit
	  bool SelectPoints_last_phase (const int x, const int y, Coords &pos, const unsigned short *const clr_matrix);

  private:
	  // fills arrays 'fv_sel_cube_x', 'fv_sel_cube_y' and 'fv_sel_cube_vis':
	  // 'fv_sel_cube_x' will get x-coordinates of all 8 corners of hit "point", 
	  // 'fv_sel_cube_y' will get y-coordinates of all 8 corners of hit "point",
	  // 'fv_sel_cube_vis' will get visibility of all 8 corners of hit "point";
	  // 'ray_pos_x', 'ray_pos_y' and 'ray_pos_z' are coordinates of rays' origin in bounding box coordinate system
	  void BuildSelectionCube (const float ray_pos_x, const float ray_pos_y, const float ray_pos_z) const;

  public:
	  // turns off "point selection" mode
	  void TurnOffPtSelection (void) { pt_selection_mode=false; }
};

