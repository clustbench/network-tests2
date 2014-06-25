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

#include "ui_matrixviewer.h"
#include <qwt_plot_spectrogram.h>
#include "../core/cntrlr_abstract.h"
#include <qwt_plot_curve.h>
#include <qwt_plot_picker.h>
#include "../core/matrixraster.h"
#include <QMdiArea>
#include <QMdiSubWindow>

class MatrixViewer: public QWidget {
	Q_OBJECT

  private:
	  Ui::ui_MatrixViewer *ui;
	  QwtPlotSpectrogram* _data[2];
	  ICntrlr *const _cntrl;
	  QwtPlotCurve *cursor;
	  QwtPlotCurve *selection_rect;
	  QwtPlotPicker *zoomer;
	  
	  QPoint _length; // lengths for matrix: from and to; if from==to then matrix will be shown else row or column
	  QPoint _p_from; // first point of matrix
	  QPoint _p_to; // last point of matrix
	  
	  const int inv; // matrix invariant (row or column); equals to (-1) when the matrix is built by message length

  public:
	  // constructor
	  MatrixViewer (QMdiArea *par, ICntrlr *cntrl, const int inva): QWidget(par), _cntrl(cntrl), inv(inva) {
		  ui=NULL;
		  _data[0]=_data[1]=NULL;
		  cursor=selection_rect=NULL;
		  zoomer=NULL;
	  }
	  
	  // must be called once after constructor;
	  // 'data[0]' must NOT be NULL and 'data' must match '_cntrl'!
	  void Init (const QString &title, MatrixRaster* data[2]);
	  
	  // destructor
	  ~MatrixViewer ();
	  
	  void SetLength (const QPoint &len);
	  void SetPointFrom (const QPoint &pnt);
	  void SetPointTo (const QPoint &pnt);
	  
	  void IAmActivated (void) { // class 'TabViewer' uses this
		  GetRowAndCol();
		  emit GiveInvariant(inv);
	  }

  protected:
	  virtual void changeEvent (QEvent *e) {
		  switch (e->type())
		  {
			case QEvent::LanguageChange:
				ui->retranslateUi(this);
				break;
			default:
				QWidget::changeEvent(e);
				break;
		  }
	  }
	  
	  virtual void closeEvent (QCloseEvent*) { emit Closing(this); }

  private:
	  void SetInfo ();

  Q_SIGNALS:
	  void GiveInvariant (const int); // class 'TabViewer' uses this
	  void Closing (QWidget*); // class 'TabViewer' uses this
	  
	  void ZoomMatrix (MatrixViewer*);
	  void RowChng (int val);
	  void ColChng (int val);

  public Q_SLOTS:
	  void SetNormalizeToWin (const bool val=true) {
		  ui->RB_normalizeCurrWindow->setEnabled(val);
		  ui->RB_normalizeLocal->setChecked(!val);
	  }
	  
	  void GetRowAndCol (void) {
		  emit RowChng(ui->SB_yFrom->value());
		  emit ColChng(ui->SB_xFrom->value());
	  }

  private Q_SLOTS:
	  void SetRightSldrMinVal (const double val);
	  void SetLeftSldrMaxVal (const double val);
	  void ShowInfo ();
	  void RectSelected (const QwtDoubleRect &rect);
	  void SetAim ();
	  void DrawSelectionRect ();
	  void ShowZoom ();
};

