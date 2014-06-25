#pragma once

#include "cntrlr_abstract.h"

class CntrlrComparison: public ICntrlr {
	Q_OBJECT
    
  private:
	  IData *source2; // second file with values

  public:
  	  // constructor
  	  //
  	  // 'err' can take one of the following values:
  	  //    NV::Success, NV::IncmpDat1Dat2
	  CntrlrComparison (IData *data1, IData *data2, NV::ErrCode &err);
	  
	  // destructor
	  virtual ~CntrlrComparison () { delete source2; }
	  
	  virtual void GetInfo (QString&) const;
	  virtual void GetMatrixRaster (const int mes_len, MatrixRaster* &res1, MatrixRaster* &res2);
	  virtual void GetRowRaster (const int row, MatrixRaster* &res1, MatrixRaster* &res2) const;
	  virtual void GetColRaster (const int col, MatrixRaster* &res1, MatrixRaster* &res2) const;
	  
	  virtual void GetPairRaster (const int row, const int col, double*&, double*&, double*&, unsigned int&) const;
	  
	  virtual char GetType () const { return 2; } // type of controller: 'comparison' (2)
	  
	  virtual QwtColorMap* AllocMainCMap (const double from, const double to) const; // don't forget to call FreeMainCMap()
	  virtual QwtColorMap* AllocAuxCMap () const { return NULL; }
	  virtual void FreeAuxCMap (QwtColorMap*) const {}
	  
	  // loads a set of matrices corresponding to message lengths from 'len_from' to 'len_to' (including 'len_to')
	  virtual NV::ErrCode SetWindow (const int len_from, const int len_to);
};

