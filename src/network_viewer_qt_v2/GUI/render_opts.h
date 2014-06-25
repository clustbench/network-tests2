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

#include <QMessageBox>

class FullViewer;
class QExpandBox;
class QRadioButton;
class QPushButton;
class QLabel;
class QSpinBox;
class QSlider;
class QLineEdit;
class QCheckBox;

class RenderOpts: public QWidget {
	Q_OBJECT

  private:
	  FullViewer *parent;
	  QExpandBox *tabs;
	  QRadioButton *repr_cubes,*repr_spheres,*repr_lights;
	  QPushButton *repr_spheres_info,*repr_lights_info;
	  unsigned int depth_ctr_ind; // index of 'depth constraint' tab
	  QSpinBox *depth_constr;
	  QPushButton *depth_info;
	  unsigned int clr_stch_ind; // index of 'color stretching' tab
	  QSlider *clr_stretching_min,*clr_stretching_max;
	  QLabel *l_clr_stretch1,*l_clr_stretch2;
	  QLineEdit *clr_stch_min,*clr_stch_max;
	  QPushButton *clr_stch_info;
	  unsigned int vol_bld_ind; // index of 'volume building' tab
	  QCheckBox *vol_bld_mode;
	  QSlider *vol_building_min1,*vol_building_max1,*vol_building_min2,*vol_building_max2;
	  QLabel *l_vol_build01,*l_vol_build02;
	  QLabel *line1,*line2;
	  QLabel *l_vol_build11,*l_vol_build12,*l_vol_build21,*l_vol_build22;
	  QLineEdit *vol_bld_min1,*vol_bld_max1,*vol_bld_min2,*vol_bld_max2;
	  QPushButton *vol_bld_info;

  public:
	  RenderOpts (const int my_width, const int working_mode, bool &was_error);
	  
	  // must be called once after the constructor
	  void Init (FullViewer *par);
	  
	  ~RenderOpts ();

  private Q_SLOTS:
	  void ChangePtRepr (void) const;
	  
	  void ShowReprSpheresInfo (void) {
		  QMessageBox::information(this,tr("About \"spheres, type 1\" representation"),
								   tr("\"Points\" are represented as spheres<br>"
								   	  "with constant intensities"));
	  }
	  
	  void ShowReprLightsInfo (void) {
		  QMessageBox::information(this,tr("About \"spheres, type 2\" representation"),
								   tr("\"Points\" are represented as spheres<br>"
								   	  "with their intensities descending<br>from "
								   	  "centres to borders (~ 1/r<sup>2</sup>)"));
	  }
	  
	  void SetDepthConstraint (void) const;
	  
	  void ShowDepthConstraintInfo (void) {
		  QMessageBox::information(this,tr("About \"depth constraint\""),
		  						   tr("\"Depth constraint\" means maximum<br>"
		  						      "number of \"points\" which can intersect<br>with one virtual ray"));
	  }
	  
	  void ShowClrStretchingInfo (void) {
		  QMessageBox::information(this,tr("About color stretching"),
								   tr("Normally minimum of all values in a file turns<br>"
									  "to color 0 and maximum - to color 255, and linear<br>"
									  "interpolation is performed on other values. By<br>"
									  "moving sliders below one can adjust minimum and<br>"
									  "maximum so all values less than this new minimum<br>"
									  "value will turn to color 0 and all values greater<br>"
									  "than this maximum value will turn to color 255<br>"
									  "(linear interpolation algorithm is left unchanged)."));
	  }
	  
	  void AdjustClrStretchingMin (const int);
	  void AdjustClrStretchingMax (const int);
	  
	  void ShowVolBuildingInfo (void) {
		  QMessageBox::information(this,tr("About volume building"),
								   tr("Only \"points\" with values between specified<br>minimum and maximum will "
									  "be visible<br><br><b>'Depth constraint' will take<br>no effect in this mode!"
									  "</b>"));
	  }
	  
	  void EnterVolBuildingMode (const int);
	  
	  void AdjustVolBuildingMin1 (const int);
	  void AdjustVolBuildingMax1 (const int);
	  void AdjustVolBuildingMin2 (const int);
	  void AdjustVolBuildingMax2 (const int);

  public:
	  void ActivateAll (void);
	  
	  void GetClrStretchingMinMax (double&, double&);
};

