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

#include <QMdiSubWindow>
#include <QLayout>
#include <QMdiArea>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>
#include <qwt_legend_item.h>
#include <qwt_text.h>

class PairViewer: public QMdiSubWindow {
	Q_OBJECT

  private:
	  QwtPlot plot;
	  QwtPlotCurve curve1,curve2,curve3;
	  QwtLegend legend;
	  QwtLegendItem leg_item1,leg_item2,leg_item3;
	  
	  Q_DISABLE_COPY(PairViewer);

  public:
	  explicit PairViewer (QWidget *parent): QMdiSubWindow(parent) {}
	  
	  void Init (const QString &title, const double *x_points, const double *y_points, const double *y_points_aux, 
				 const unsigned int num_points, const char type) {
		  QPen pen;
		  pen.setWidth(2);
		  
		  this->setWindowTitle(title);
		  this->resize(parentWidget()->size());
		  
		  plot.setParent(this);
		  legend.setParent(this);
		  legend.hide();
		  
		  curve1.setData(x_points,y_points,num_points);
		  curve1.setPen(pen);
		  
		  plot.enableAxis(QwtPlot::xBottom,true);
		  plot.setAxisTitle(QwtPlot::xBottom,tr("message<br>lengths")); // Ox
		  plot.enableAxis(QwtPlot::yLeft,true);
		  plot.setAxisTitle(QwtPlot::yLeft,tr("values")); // Oy
		  
		  curve1.attach(&plot);
		  
		  if (y_points_aux!=NULL)
		  {
			  // add curve for values in the second file
			  
			  QwtText text;
			  
			  if (type==1)
			  {
				  curve1.setTitle(tr("values"));
				  text.setText(tr("deviations"));
				  text.setColor(Qt::red);
				  curve2.setTitle(text);
			  }
			  else
			  {
				  text.setText(tr("values<br>in file 1"));
				  text.setColor(Qt::green);
				  curve1.setTitle(text);
				  text.setText(tr("values<br>in file 2"));
				  text.setColor(Qt::red);
				  curve2.setTitle(text);
				  
				  // green curve for values in the first file
				  pen.setColor(Qt::green);
				  curve1.setPen(pen);
				  
				  // brown curve for difference of values in files
				  double *dfr=static_cast<double*>(malloc(num_points*sizeof(double)));
				  for (unsigned int i=0u; i<num_points; ++i)
					  dfr[i]=y_points[i]-y_points_aux[i];
				  curve3.setData(x_points,dfr,num_points);
				  curve3.attach(&plot);
				  pen.setColor(Qt::darkYellow);
				  curve3.setPen(pen);
				  text.setText(tr("difference<br>between<br>(1) and (2)"));
				  text.setColor(Qt::darkYellow);
				  curve3.setTitle(text);
				  free(dfr);
			  }
			  // red curve for values in the second file
			  curve2.setData(x_points,y_points_aux,num_points);
			  pen.setColor(Qt::red);
			  curve2.setPen(pen);
			  curve2.attach(&plot);
			  
			  leg_item1.setIdentifierMode(QwtLegendItem::ShowText);
			  legend.insert(&curve1,&leg_item1);
			  leg_item2.setIdentifierMode(QwtLegendItem::ShowText);
			  legend.insert(&curve2,&leg_item2);
			  if (type==2)
			  {
				  leg_item3.setIdentifierMode(QwtLegendItem::ShowText);
				  legend.insert(&curve3,&leg_item3);
			  }
			  legend.show();
			  plot.insertLegend(&legend,QwtPlot::RightLegend);
		  }
		  
		  this->layout()->addWidget(&plot);
		  
		  plot.replot();
	  }
	  
	  Q_SIGNAL void Closing (QWidget*); // class 'TabViewer' uses this

  protected:
	  virtual void closeEvent (QCloseEvent*) { emit Closing(this);  }
};

