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

#include "renderer.h"
#include <cmath>
#include <cstdlib>
#include <QString>

/* OpenMP-based renderer */
class RendererOMP: public Renderer {    
  private:
	  unsigned int img_w,img_h; // image width and height (will be received from FullViewer)
	  
	  /* main box rendering (with transparency) */
	  unsigned short *matrix; // color matrix
	  int x_num,y_num,z_num; // 'matrix' dimensions' sizes (number of points)
	  float point_step; // distance between two "points"
	  float point_radius; // "point" radius
	  float cam_x,cam_y,cam_z; // "camera" position
	  
	  /* bounding box of all "points" */
	  float x_size,y_size,z_size; // real edges' sizes
	  float bbox_cnt_x,bbox_cnt_y,bbox_cnt_z; // centre point
	  float ox_x,ox_y,ox_z,oy_x,oy_y,oy_z,oz_x,oz_y,oz_z; // axes of the box (normalized!)
	  
	  int depth_constraint; // maximum number of "points" which can intersect with one ray
	  
	  /* color constraints */
	  unsigned int clr_minmax; // bit field; contains max_red | min_red | max_green | min_green
	  
	  void (RendererOMP::*RendBox)(unsigned int*) const; // pointer to main render function (according 
	  													 // to representation of "points")
	  
	  /* "point selection" */
	  bool pt_selection_mode;
	  int sel_ind_x,sel_ind_y,sel_ind_z; // coordinates of selected "point" 
	  									 // (are valid only during "point selection" mode!)
	  int *const fv_sel_cube_x,*const fv_sel_cube_y; // coordinates of selection cube
	  bool *const fv_sel_cube_vis; // visibility of selection cube's points
	  
	  QString err_str; // contains error messages

  public:
	  // constructor
	  RendererOMP (int *const fv_sl_cube_x, int *const fv_sl_cube_y, bool *const fv_sl_cube_vis): 
	    RendBox(&RendererOMP::RenderBox_cubes), fv_sel_cube_x(fv_sl_cube_x), fv_sel_cube_y(fv_sl_cube_y), 
	    fv_sel_cube_vis(fv_sl_cube_vis) {}
	  
	  // must be called once after constructor
	  void Init (const unsigned int image_w, const unsigned int image_h, 
	  			 const int num_x, const int num_y, const int num_z);
	  
	  // returns error string
	  const QString& GetError (void) const { return err_str; }
	  
	  // returns the number of compute units on the device
	  unsigned int NumberCUs (void) const;
	  
	  // gets 'clr_matrix' variable from FullViewer
	  void SetClrMatrix (unsigned short *clr_matrix) { matrix=clr_matrix; }
	  
	  // shifts "camera" origin by vector {sh_x,sh_y,sh_z}
	  void ShiftCamera (const float sh_x, const float sh_y, const float sh_z) {
		  cam_x+=sh_x;
		  cam_y+=sh_y;
		  cam_z+=sh_z;
		  if (pt_selection_mode)
			  BuildSelectionCube((cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+
								 ((float)x_num)*point_step*0.5f,
								 (cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+
								 ((float)y_num)*point_step*0.5f,
								 (cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+
								 ((float)z_num)*point_step*0.5f);
	  }
	  
	  // increases/decreases bounding box size (according to 'inc') by increasing/decreasing 'point_step'
	  bool Zoom (const bool inc) {
		  const float new_point_step=point_step+(inc? 0.5f : (-0.5f));
		  if (new_point_step>0.0f)
		  {
			  const float koeff=new_point_step/point_step;
			  
			  x_size*=koeff;
			  y_size*=koeff;
			  z_size*=koeff;
			  
			  point_radius*=koeff;
			  
			  point_step=new_point_step;
			  
			  if (pt_selection_mode)
				  BuildSelectionCube((cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+
									 ((float)x_num)*point_step*0.5f,
									 (cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+
									 ((float)y_num)*point_step*0.5f,
									 (cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+
									 ((float)z_num)*point_step*0.5f);
			  return true;
		  }
		  return false;
	  }
	  
	  // rotates bounding box by 'w_angle' in OXZ and 'h_angle' in OYZ
	  void Rotate (const float w_angle, const float h_angle) {
		  // rotation in OXZ
		  float sina=sinf(w_angle),cosa=cosf(w_angle);
		  float tmp=ox_x;
		  ox_x=ox_x*cosa-ox_z*sina;
		  ox_z=tmp*sina+ox_z*cosa;
		  tmp=oy_x;
		  oy_x=oy_x*cosa-oy_z*sina;
		  oy_z=tmp*sina+oy_z*cosa;
		  tmp=oz_x;
		  oz_x=oz_x*cosa-oz_z*sina;
		  oz_z=tmp*sina+oz_z*cosa;
		  
		  // rotation in OYZ
		  sina=sinf(h_angle);
		  cosa=cosf(h_angle);
		  tmp=ox_y;
		  ox_y=ox_y*cosa-ox_z*sina;
		  ox_z=tmp*sina+ox_z*cosa;
		  tmp=oy_y;
		  oy_y=oy_y*cosa-oy_z*sina;
		  oy_z=tmp*sina+oy_z*cosa;
		  tmp=oz_y;
		  oz_y=oz_y*cosa-oz_z*sina;
		  oz_z=tmp*sina+oz_z*cosa;
		  
		  if (pt_selection_mode)
			  BuildSelectionCube((cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+
								 ((float)x_num)*point_step*0.5f,
								 (cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+
								 ((float)y_num)*point_step*0.5f,
								 (cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+
								 ((float)z_num)*point_step*0.5f);
	  }
	  
	  // rotates bounding box by arccos('cosa') in OXY
	  void RotateOXY (const float cosa, const bool positive) {
		  const float sina=sqrtf(1.0f-cosa*cosa);
		  float tmp=ox_x;
		  
		  if (positive)
		  {
			  ox_x=tmp*cosa-ox_y*sina;
			  ox_y=tmp*sina+ox_y*cosa;
			  tmp=oy_x;
			  oy_x=tmp*cosa-oy_y*sina;
			  oy_y=tmp*sina+oy_y*cosa;
			  tmp=oz_x;
			  oz_x=tmp*cosa-oz_y*sina;
			  oz_y=tmp*sina+oz_y*cosa;
		  }
		  else
		  {
			  ox_x=tmp*cosa+ox_y*sina;
			  ox_y=ox_y*cosa-tmp*sina;
			  tmp=oy_x;
			  oy_x=tmp*cosa+oy_y*sina;
			  oy_y=oy_y*cosa-tmp*sina;
			  tmp=oz_x;
			  oz_x=tmp*cosa+oz_y*sina;
			  oz_y=oz_y*cosa-tmp*sina;
		  }
		  if (pt_selection_mode)
			  BuildSelectionCube((cam_x-bbox_cnt_x)*ox_x+(cam_y-bbox_cnt_y)*ox_y+(cam_z-bbox_cnt_z)*ox_z+
								 ((float)x_num)*point_step*0.5f,
								 (cam_x-bbox_cnt_x)*oy_x+(cam_y-bbox_cnt_y)*oy_y+(cam_z-bbox_cnt_z)*oy_z+
								 ((float)y_num)*point_step*0.5f,
								 (cam_x-bbox_cnt_x)*oz_x+(cam_y-bbox_cnt_y)*oz_y+(cam_z-bbox_cnt_z)*oz_z+
								 ((float)z_num)*point_step*0.5f);
	  }
	  
	  // writes projection on the screen of all 3 axes to 'axes';
	  // the projection is computed by simply throwing Z-coordinates away;
	  // 'axes' must have at least 6 elements:
	  //    [0],[1] - Ox;
	  //    [2],[3] - Oy;
	  //    [4],[5] - Oz
	  void GetAxesForDrawing (float *axes) const {
		  axes[0]=ox_x;
		  axes[1]=ox_y;
		  axes[2]=oy_x;
		  axes[3]=oy_y;
		  axes[4]=oz_x;
		  axes[5]=oz_y;
	  }

  private:
	  // variant of RenderBox() (see below). "Points" are represented as cubes
	  void RenderBox_cubes (unsigned int *pixels) const;
	  
	  // variant of RenderBox() (see below). "Points" are represented as spheres
	  void RenderBox_spheres (unsigned int *pixels) const;
	  
	  // variant of RenderBox() (see below). "Points" are represented as so called "lights"
	  // (they are spheres but their intensity descends from centre to borders)
	  void RenderBox_lights (unsigned int *pixels) const;
	  
	  // variant of RenderBox() (see below). Builds a volume; "points" are represented as cubes
	  void RenderBox_cubes_vol (unsigned int *pixels) const;
	  
	  // variant of RenderBox() (see below). Builds a volume; "points" are represented as spheres
	  void RenderBox_spheres_vol (unsigned int *pixels) const;
	  
	  // variant of RenderBox() (see below). Builds a volume; "points" are represented as so called "lights"
	  // ("lights" are spheres but their intensity descends from centre to borders)
	  void RenderBox_lights_vol (unsigned int *pixels) const;
	  
	  // fills arrays 'fv_sel_cube_x', 'fv_sel_cube_y' and 'fv_sel_cube_vis':
	  // 'fv_sel_cube_x' will get x-coordinates of all 8 corners of hit "point", 
	  // 'fv_sel_cube_y' will get y-coordinates of all 8 corners of hit "point",
	  // 'fv_sel_cube_vis' will get visibility of all 8 corners of hit "point";
	  // 'ray_pos_x', 'ray_pos_y' and 'ray_pos_z' are coordinates of rays' origin in bounding box coordinate system
	  void BuildSelectionCube (const float ray_pos_x, const float ray_pos_y, const float ray_pos_z) const;

  public:
	  /// main render function
	  void RenderBox (unsigned int *pixels) const { (this->*RendBox)(pixels); }
	  
	  // chooses variants of RenderBox according to 'pt_repr'
	  PtRepr ChangeKernel (const PtRepr pt_repr) {
		  this->err_str.clear();
		  switch (pt_repr)
		  {
			case CUBES:
				if (RendBox==&RendererOMP::RenderBox_cubes) return CUBES;
				RendBox=&RendererOMP::RenderBox_cubes;
				return SPHERES; // return something that differs from 'CUBES'
			case SPHERES:
				if (RendBox==&RendererOMP::RenderBox_spheres) return SPHERES;
				RendBox=&RendererOMP::RenderBox_spheres;
				return LIGHTS; // return something that differs from 'SPHERES'
			case LIGHTS:
				if (RendBox==&RendererOMP::RenderBox_lights) return LIGHTS;
				RendBox=&RendererOMP::RenderBox_lights;
				return CUBES; // return something that differs from 'LIGHTS'
		  }
		  return CUBES; // we must not be here
	  }
	  
	  // sets new value for 'depth_constraint'
	  void SetDepthConstraint (const unsigned int new_d_c) { depth_constraint=static_cast<int>(new_d_c); }
	  
	  // returns 'true' in "volume building" mode
	  bool IsVolumeMode (void) const {
		  return ((RendBox==&RendererOMP::RenderBox_cubes_vol) || (RendBox==&RendererOMP::RenderBox_spheres_vol) || 
				  (RendBox==&RendererOMP::RenderBox_lights_vol));
	  }
	  
	  // turns on/off "volume building" mode;
	  bool ToggleVolumeMode (const bool on) {
		  this->err_str.clear();
		  if (on)
		  {
			  if (RendBox==&RendererOMP::RenderBox_cubes) RendBox=&RendererOMP::RenderBox_cubes_vol;
			  else
				  if (RendBox==&RendererOMP::RenderBox_spheres) RendBox=&RendererOMP::RenderBox_spheres_vol;
				  else
					  // do NOT remove this condition!
					  if (RendBox==&RendererOMP::RenderBox_lights) RendBox=&RendererOMP::RenderBox_lights_vol;
		  }
		  else
		  {
			  if (RendBox==&RendererOMP::RenderBox_cubes_vol) RendBox=&RendererOMP::RenderBox_cubes;
			  else
			  	  if (RendBox==&RendererOMP::RenderBox_spheres_vol) RendBox=&RendererOMP::RenderBox_spheres;
			  	  else
			  	  	  // do NOT remove this condition!
			  	  	  if (RendBox==&RendererOMP::RenderBox_lights_vol) RendBox=&RendererOMP::RenderBox_lights;
		  }
		  return true;
	  }
	  
	  // removes "points" with green color not in [min_green,max_green] and red color not in [min_red,max_red]
	  // must be called ONLY after ToggleVolumeMode(true)!!
	  bool BuildVolume (const unsigned char min_green, const unsigned char max_green, 
						const unsigned char min_red, const unsigned char max_red) {
		  clr_minmax=((unsigned int)min_green) | (((unsigned int)max_green)<<8) | 
					  (((unsigned int)min_red)<<16) | (((unsigned int)max_red)<<24);
		  this->err_str.clear();
		  return true;
	  }
	  
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
	  
	  // turns off "point selection" mode
	  void TurnOffPtSelection (void) { pt_selection_mode=false; }
};

