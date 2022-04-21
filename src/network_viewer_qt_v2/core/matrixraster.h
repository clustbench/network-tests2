#pragma once

#include "data_abstract.h"
#include <qwt_raster_data.h>
#include <qwt_interval.h>

class MatrixRaster: public QwtRasterData {
  private:
  	  mutable double *data_array; // data matrix
  	  const int rows,cols; // sizes of 'data_array'
	  double from,to; // minimum and maximum values in 'data_array'
	
	  Q_DISABLE_COPY(MatrixRaster)

  public:
	  MatrixRaster (double* &matr, const int rws, const int cls); // 'matr' will be "stolen"
	
	  ~MatrixRaster () { if (data_array!=NULL) free(data_array); }
	
	  virtual QwtRasterData* copy () const { // derived from QwtRasterData
		  MatrixRaster *new_m_r=new MatrixRaster(data_array,rows,cols);
		  data_array=NULL;
		  return new_m_r;
	  }
	
	  virtual QwtInterval range () const { return QwtInterval(from,to); } // derived from QwtRasterData
	
	  virtual double value (double col, double row) const override{ // derived from QwtRasterData
		  int x=static_cast<int>(col),y=static_cast<int>(row);
		  if ((x>=0) && (x<cols) && (y>=0) && (y<rows)) return data_array[y*cols+x];
		  return 0.0;
	  }
	
	  QwtInterval interval( Qt::Axis ) const override { return QwtInterval(from,to); }  //must be derived because it is pure virtual in the base class

	  const double* Data () const { return data_array; }
	
	  int GetRows () const { return rows; }
	  int GetCols () const { return cols; }
};

