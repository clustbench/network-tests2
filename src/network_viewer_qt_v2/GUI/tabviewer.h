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

#include <QMainWindow>
#include "ui_tabviewer.h"
#include "../core/cntrlr_abstract.h"
#include "matrixviewer.h"
#include "mainwindow.h"

class TabViewer: public QMainWindow {
	Q_OBJECT

  private:
  	  static const QString my_sign; // sign for log messages
  	  ICntrlr *const controller;
	  Ui::ui_TabViewer *ui;

  private:
	  // constructor
	  TabViewer (ICntrlr *cntrlr, QMainWindow *parent): QMainWindow(parent), controller(cntrlr) {
		  ui=NULL;
	  }
	  
	  // must be called once after constructor
	  bool Init (void);
	  
	  Q_DISABLE_COPY(TabViewer);

  public:
	  // replaces calls to both the constructor and Init() (it was designed to ensure a call to Init()); 
	  // returns 'NULL' if any errors occured; 
	  // returned pointer should be deleted using 'delete'
	  static TabViewer* Create (ICntrlr *cntrlr, QMainWindow *parent) {
		  TabViewer *new_tv;
		  
		  try {
			  new_tv=new TabViewer(cntrlr,parent);
		  }
		  catch (...) {
			  return NULL;
		  }
		  if (!new_tv->Init())
		  {
			  delete new_tv;
			  return NULL;
		  }
		  return new_tv;
	  }
	  
	  // destructor
	  ~TabViewer () {
		  if (controller!=NULL) delete controller;
		  if (ui!=NULL) delete ui;
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

  Q_SIGNALS:
	  void SendMessToLog (const MainWindow::MsgType, const QString &msg, const QString &stat);

  private Q_SLOTS:
	  void Initialize ();
	  
	  void ShowMesLen ();
	  void ShowRow ();
	  void ShowCol ();
	  void ShowPair ();
	  
	  void LoadWindow ();
	  void DropWindow ();
	  
	  void ChangeMatrNumber (const double val) { ui->SB_MatrixNumber->setValue(val); }
	  
	  void ChangeLoadWindowBtn (void) {
	  	  ui->B_LoadWindow->setEnabled(ui->SB_LoadWinFrom->value()<ui->SB_LoadWinTo->value());
	  }
	  
	  void DeleteSubWindow (QWidget *sub) {
		  ui->mdiArea->removeSubWindow(sub);
		  delete sub;
	  }
	  
	  void SubActivated (QMdiSubWindow *sub) const {
		  if (sub!=NULL)
		  {
			  MatrixViewer *const msub=dynamic_cast<MatrixViewer*>(sub->widget());
			  if (msub!=NULL) msub->IAmActivated();
		  }
	  }

  public Q_SLOTS:
	  void SetProgressBarValue (const int val) {
		  if (val==-1) ui->progressBar->hide();
		  else
		  {
			  if (!ui->progressBar->isVisible()) ui->progressBar->show();
			  ui->progressBar->setValue(val);
		  }
	  }
	  
	  void SetRowValue (const int val) {
		  if (ui->CB_updFromCurrMtr->isChecked() /*&& ui->SB_row->isEnabled()*/)
			  ui->SB_row->setValue(val);
	  }
	  
	  void SetColValue (const int val) {
		  if (ui->CB_updFromCurrMtr->isChecked() /*&& ui->SB_column->isEnabled()*/)
			  ui->SB_column->setValue(val);
	  }
	  
	  void NewMatrix_mes (MatrixViewer*);
	  void NewMatrix_row (MatrixViewer*);
	  void NewMatrix_col (MatrixViewer*);
};

