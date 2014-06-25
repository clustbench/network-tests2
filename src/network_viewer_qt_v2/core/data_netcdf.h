#pragma once

#include "data_abstract.h"

/* Class to process input files in NetCDF format */
class Data_NetCDF: public IData {
  private:
	  int source_file; // NetCDF descriptor of source file
	  int data_var; // NetCDF descriptor of main data in the file
	  
	  /* data for Begin()-IsEnd()-GetDataAndMove() */
	  size_t start[3],count[3];
	  IData::Portion portion;

  private:
	  Data_NetCDF (const Data_NetCDF&); // denied!!
	  void operator= (const Data_NetCDF&); // denied!!

  public:
	  // constructor
	  //
	  // 'err' can take one of the following values:
	  //    NV::Success;
	  //    NV::NotANetCDF, NV::NoNumProc, NV::NoBegMesLen, NV::NoEndMesLen, 
	  //    NV::NoStepLen, NV::NoNoiseMesLen, NV::NoNoiseMesNum, NV::NoNoiseNumProc, 
	  //    NV::NoRpts, NV::NoHosts, NV::No3DData, NV::NoMem
	  Data_NetCDF (const QString &f_name, const QString &hosts_f_name, NV::ErrCode &err);
	  
	  // destructor
	  ~Data_NetCDF ();
	  
	  /* 3 functions to read the whole file */
	  // sets the portion and moves to the determined position in the file
	  virtual void Begin (const IData::Portion, const int mes_len);
	  // returns 'true' if the file has reached its end
	  virtual bool IsEnd (void) const { return (start[0]==static_cast<size_t>(this->z_num)); }
	  // reads data to 'buf' and moves toward next portion
	  virtual NV::ErrCode GetDataAndMove (double *buf);
	  
	  // gets single value;
	  // {'row','col','mes_len'} means {x,y,z} of the retrieved value and may be explained such that:
	  //    we reach the matrix corresponding to message length 'mes_len', then read rows 
	  //    consequently until the row number 'row'; from this row we get value number 'col'
	  virtual NV::ErrCode GetSingleValue (const int mes_len, const int row, const int col, double&);
};

