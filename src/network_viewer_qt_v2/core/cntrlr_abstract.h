#pragma once

#include "data_abstract.h"
#include <qwt_color_map.h>

class MatrixRaster;

class ICntrlr: public QObject {
	Q_OBJECT

  protected:
	  IData *source;
	  double **window; // a set of matrices corresponding to several subsequent message lengths
	  unsigned int window_size;
	  int window_borders[2]; // [0] - lower, [1] - higher

  public:
	  ICntrlr (IData *data) {
		  source=data;
		  window=NULL;
		  window_size=0u;
		  window_borders[0]=0;
		  window_borders[1]=0;
	  }
	  
	  virtual ~ICntrlr () {
		  DropWindow();
		  delete source;
	  };
	  
  	  // collects test parameters in 'source' and writes them to string 'info'
	  virtual void GetInfo (QString &info) const = 0;
	  
	  void GetHosts (QString &hosts) const {
	  	  source->GetHostNamesAsString(hosts);
	  	  if (hosts.isEmpty())
			  hosts=tr("Hosts' names are unavailable");
	  }
	  
	  virtual void GetMatrixRaster (const int mes_len, MatrixRaster* &res1, MatrixRaster* &res2) = 0;
	  virtual void GetRowRaster (const int row, MatrixRaster* &res1, MatrixRaster* &res2) const = 0;
	  virtual void GetColRaster (const int col, MatrixRaster* &res1, MatrixRaster* &res2) const = 0;
	  virtual void GetPairRaster (const int row, const int col, double*&, double*&, double*&, unsigned int&) const = 0;
	  
	  const QString& GetSourceFileName (void) const { return source->GetSourceFileName(); }
	  
	  int GetNumProcessors () const { return source->GetNumProcessors(); }
	  int GetBeginMessageLength () const { return source->GetBeginMessageLength(); }
	  int GetRealEndMessageLength () const { return source->GetRealEndMessageLength(); }
	  int GetStepLength () const { return source->GetStepLength(); }
	  const int* GetWindowBorders () const { return window_borders; }
	  
	  virtual char GetType () const = 0; // type of controller: 'single' (0) , 'deviation' (1) or 'comparison' (2)
	  
	  virtual QwtColorMap* AllocMainCMap (const double from, const double to) const = 0; // don't forget to call FreeMainCMap()
	  void FreeMainCMap (QwtColorMap *m_cmap) const { delete m_cmap; }
	  virtual QwtColorMap* AllocAuxCMap () const = 0; // don't forget to call FreeAuxCMap()
	  virtual void FreeAuxCMap (QwtColorMap *a_cmap) const = 0;
	  
	  // loads a set of matrices corresponding to message lengths from 'len_from' to 'len_to' (including 'len_to')
	  virtual NV::ErrCode SetWindow (const int len_from, const int len_to) = 0;
	  // deletes previously loaded matrices
	  void DropWindow (void) {
		  if (window!=NULL/*window_size!=0u*/)
		  {
			  for (unsigned int i=0u; i<window_size; ++i)
				  free(window[i]);
			  free(window);
			  window=NULL;
		  }
	  }
	  
  Q_SIGNALS:
	  void Progress (const int val);
};

