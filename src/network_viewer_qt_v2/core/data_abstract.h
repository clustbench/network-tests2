#pragma once

#include <QString>
#include <QVector>
#include "err_codes.h"

class IData {
  public:
	  /* file types */
	  typedef enum { NetCDF, Txt } Type;
	  
	  /* portion of data which will be read each time */
	  typedef enum { File, Matrix, Row, Value } Portion;

  protected:
	  QString source_fname; // full name of input file
	  
	  /* test parameters; obtained from input file only one time */
	  QString test_type;
	  QString data_type;
	  int num_processors;
	  int begin_message_length;
	  int end_message_length; // unreliable value, only for showing to the user;
	  						  // as I understood it means some length which is GREATER then 
	  						  // the last valid length (such length that a matrix exists for it);
	  						  // use GetRealEndMessageLength() instead
	  int step_length;
	  int noise_message_length;
	  int noise_message_num;
	  int noise_processors;
	  int num_repeats;
	  
	  int z_num;
	  
	  QVector<QString> host_names; // array of enumerated hosts' names (from 1); names contain '\n'
	  
  public:
	  // we need virtual destructor
	  virtual ~IData () {}
	  
	  const QString& GetSourceFileName () const { return source_fname; }
	  
	  /* get various test parameters */
	  const QString& GetTestType () const { return test_type; }
	  const QString& GetDataType () const { return data_type; }
	  int GetNumProcessors () const { return num_processors; }
	  int GetBeginMessageLength () const { return begin_message_length; }
	  int GetEndMessageLength () const { return end_message_length; }
	  int GetStepLength () const { return step_length; }
	  int GetNoiseMessageLength () const { return noise_message_length; }
	  int GetNoiseMessageNum () const { return noise_message_num; }
	  int GetNoiseProcessors () const { return noise_processors; }
	  int GetNumRepeats () const { return num_repeats; }
	  
	  int GetZNum () const { return z_num; }
	  
	  // gets the last valid message length
	  int GetRealEndMessageLength () const { return begin_message_length+(z_num-1)*step_length; }
	  
	  // returns vector with hosts' names
	  const QVector<QString>& GetHostNamesAsVector (void) { return host_names; }
	  // returns one long string with hosts' names
	  void GetHostNamesAsString (QString &str) const {
		  QVector<QString>::const_iterator it,it_end;
		  
		  str.clear();
		  try {
			  for (it=host_names.constBegin(),it_end=host_names.constEnd(); it!=it_end; ++it)
				  str+=*it;
		  }
		  catch (...) { str.clear(); } // not enough memory
	  }
	  
	  /* 3 functions to read the whole file */
	  // sets the portion and moves to the determined position in the file
	  virtual void Begin (const Portion, const int mes_len) = 0;
	  // returns 'true' if the file has reached its end
	  virtual bool IsEnd (void) const = 0;
	  // reads data to 'buf' and moves toward next portion
	  virtual NV::ErrCode GetDataAndMove (double *buf) = 0;
	  
	  // gets single matrix of values according to message length;
	  // 'buf' must hold at least 'num_processors*num_processors*sizeof(double)' bytes of space
	  NV::ErrCode GetMatrix (const int mes_length, double *buf) {
	  	  Begin(Matrix,mes_length);
	  	  return GetDataAndMove(buf);
	  }
	  
	  // gets single value;
	  // {'row','col','mes_len'} means {x,y,z} of the retrieved value and may be explained such that:
	  //    we reach the matrix corresponding to message length 'mes_len', then read rows 
	  //    consequently until the row number 'row'; from this row we get value number 'col'
	  virtual NV::ErrCode GetSingleValue (const int mes_len, const int row, const int col, double&) = 0;
	  
	  bool operator == (const IData *const dt) const {
		  return ((num_processors==dt->num_processors) && (begin_message_length==dt->begin_message_length) && 
				  (z_num==dt->z_num) && (step_length==dt->step_length) && 
				  (noise_message_length==dt->noise_message_length) && (noise_message_num==dt->noise_message_num) && 
				  (noise_processors==dt->noise_processors) /*&& (host_names==dt->host_names)*/);
	  }
};

