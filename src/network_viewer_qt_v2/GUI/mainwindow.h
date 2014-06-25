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
#include <QTranslator>
#include "ui_mainwindow.h"
class QTranslator;

class MainWindow: public QMainWindow {
	Q_OBJECT
	
	/* types of messages which are printed to the log */
  public:
	typedef enum { Error, Success, Info } MsgType;
	
  private:
	  Ui::MainWindow *ui;
	  QTranslator *app_tr,*qt_tr;

  public:
  	  // constructor
	  explicit MainWindow (QWidget *parent): QMainWindow(parent) {
		  ui=NULL;
		  app_tr=qt_tr=NULL;
	  }
	  
	  // must be called once after constructor
	  void Init (void);
	  
	  // destructor
	  ~MainWindow ();
	  
	  // prints user message 'msg' to 'ui->TE_Log';
	  // 'msg' is allowed to be in rich text format;
	  // 'sign' - short string describing the sender of 'msg'; 'sign' will be shown in square brackets
	  Q_SLOT void AddMsgToLog (const /*!*/MainWindow::/*!*/MsgType, const QString &sign, const QString &msg);

  protected:
	  void changeEvent (QEvent *e) {
		  if (e->type()==QEvent::LanguageChange)
			  ui->retranslateUi(this);
		  QMainWindow::changeEvent(e);
	  }

  public Q_SLOTS:
	  void ChangeTabTitle (QWidget*, const QString&);
	  void CloseTab (QWidget*);

  private Q_SLOTS:
	  void ShowAbout (void);
	  
	  void CloseTab (const int);
	  
	  void Load (void);
	  
	  void ToggleWidgetsIn_2D_3D (const int);
	  void ToggleLog (const bool);
	  
	  void SwitchLanguageToEng (const bool);
	  void SwitchLanguageToRus (const bool);
};

