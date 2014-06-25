#include "matrixraster.h"
#include <cfloat>

MatrixRaster::MatrixRaster (double* &matr, const int rws, const int cls): 
  QwtRasterData(QwtDoubleRect(0.0,0.0,cls,rws)), data_array(matr), rows(rws), cols(cls) {  
	this->initRaster(QwtDoubleRect(0.0,1.0,0.0,1.0),QSize(cls,rws));

	if ((matr==NULL) || (rws==0) || (cls==0))
	{
		from=DBL_MAX;
		to=1.0-DBL_MAX;
	}
	else
	{
		const unsigned int matr_size=static_cast<const unsigned int>(rws)*static_cast<const unsigned int>(cls);
		double tmp;

		to=from=matr[0];
		for (unsigned int i=1u; i<matr_size; ++i)
		{
			tmp=matr[i];
			from=(tmp<from)? tmp : from;
			to=(to<tmp)? tmp : to;
		}
		matr=NULL;
	}
}

