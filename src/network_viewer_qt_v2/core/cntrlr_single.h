#pragma once

#include "cntrlr_abstract.h"

class CntrlrSingle: public ICntrlr {
	Q_OBJECT
    
  public:
	  CntrlrSingle (IData *data, NV::ErrCode &err): ICntrlr(data) {
	  	  err=NV::Success;
	  }
	  
	  virtual void GetInfo (QString&) const;
	  virtual void GetMatrixRaster (const int mes_len, MatrixRaster* &res1, MatrixRaster* &res2);
	  virtual void GetRowRaster (const int row, MatrixRaster* &res1, MatrixRaster* &res2) const;
	  virtual void GetColRaster (const int col, MatrixRaster* &res1, MatrixRaster* &res2) const;
	  virtual void GetPairRaster (const int row, const int col, double*&, double*&, double*&, unsigned int&) const;
	  
	  virtual char GetType () const { return 0; } // type of controller: 'single' (0)
	  
	  virtual QwtColorMap* AllocMainCMap (const double from, const double to) const; // don't forget to call FreeMainCMap()
	  virtual QwtColorMap* AllocAuxCMap () const { return NULL; }
	  virtual void FreeAuxCMap (QwtColorMap*) const {}
	  
	  // loads a set of matrices corresponding to message lengths from 'len_from' to 'len_to' (including 'len_to')
	  virtual NV::ErrCode SetWindow (const int len_from, const int len_to);
};

