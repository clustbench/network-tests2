#pragma once

#include "data_abstract.h"
#include <cstdio>

/* Class to process input files in text format */
class Data_Text: public IData {
  public:
  	  /* class-wrapper for use in Data_Text::readline() */
  	  class Line {
  	  	  friend class Data_Text;
  	  	  
  	  	private:
  	  		char *line; // string
  	  		unsigned int mem_sz; // size of allocated memory (in bytes) for 'line'
  	  	  	  
  	  	public:
  	  		// the only constructor
  	  		Line () { line=NULL; mem_sz=0u; }
  	  		
  	  		// frees the memory
  	  		void Destroy (void) {
  	  			if (line!=NULL)
  	  	  	  	{
	  	  	  	  	free(line);
	  	  	  	  	line=NULL;
  	  	  	  	}
  	  	  	  	mem_sz=0u;
  	  	  	}
  	  	  	
  	  	  	// destructor
  	  	  	~Line () { Destroy(); }
  	  	  	
  	  	  	// only shows contents of 'line'
  	  	  	const char* Give (void) const { return line; }
  	  	  	
  	  	  	// returns 'line' with the ability to modify it
  	  	  	char* Give_mdf (void) { return line; }
  	  	  	
  	  	  	// move 'this->line' to 'l'; 'this' becomes empty
  	  	  	//void Get (char* &l) { l=line; line=NULL; mem_sz=0u; }
  	  	  	
  	  	  	// checks if the whole 'line' consists of white space
  	  	  	bool isallws (void) const {
  	  	  		if (line==NULL) return true;
  	  	  		
  	  	  		const char *tmp=line;
  	  	  		char c;
  	  	  		
  	  	  		for ( ; ; )
  	  	  	  	{
	  	  	  	  	c=*tmp;
	  	  	  	  	if (c=='\0') break;
	  	  	  	  	if (!isspace(c)) return false;
	  	  	  	  	++tmp;
  	  	  	  	}
  	  	  	  	return true;
  	  	  	}
  	  };
  	  
  	  // reads one line ('\n' is included) from 'f';
	  // returns 'false' in case of some error
	  static bool readline (FILE *f, Line&);
	  
  private:
	  FILE *source_file;
	  fpos_t data_pos; // position in 'source_file' of the very first value in 3D data
	  char flt_pt; // decimal point character (',' or '.' in floating-point numbers)
	  
	  /* data for Begin()-IsEnd()-GetDataAndMove() */
	  int matr_ind,row_val_ind;
	  IData::Portion portion;

  private:
	  Data_Text (const Data_Text&); // denied!!
	  void operator= (const Data_Text&); // denied!!

  public:
	  // constructor
	  //
	  // 'err' can take one of the following values:
	  //    NV::Success;
	  //    NV::CannotOpen, NV::UnexpEOF, NV::NoNumProc, NV::NoBegMesLen, 
	  //    NV::NoEndMesLen, NV::NoStepLen, NV::NoNoiseMesLen, NV::NoNoiseMesNum, 
	  //    NV::NoNoiseNumProc, NV::NoRpts, NV::No3DData, NV::NoMem
	  Data_Text (const QString &f_name, NV::ErrCode &err);
	  
	  // destructor
	  ~Data_Text () { if (source_file!=NULL) fclose(source_file); }
	  
	  /* 3 functions to read the whole file */
	  // sets the portion and moves to the determined position in the file
	  virtual void Begin (const IData::Portion, const int mes_len);
	  // returns 'true' if the file has reached its end
	  virtual bool IsEnd (void) const { return (matr_ind==this->z_num); }
	  // reads data to 'buf' and moves toward next portion
	  virtual NV::ErrCode GetDataAndMove (double *buf);
	  
	  // gets single value;
	  // {'row','col','mes_len'} means {x,y,z} of the retrieved value and may be explained such that:
	  //    we reach the matrix corresponding to message length 'mes_len', then read rows 
	  //    consequently until the row number 'row'; from this row we get value number 'col'
	  virtual NV::ErrCode GetSingleValue (const int mes_len, const int row, const int col, double&);
};

