#include "cntrlr_single.h"
#include "matrixraster.h"

void CntrlrSingle::GetInfo (QString &info) const {
	const IData *src=this->source;
	info=tr("<p><b>File path:</b> %1<br>"
			"<br><b>Processors:</b> %2<br>"
			"<br><b>Test type:</b> %3<br>"
			"<br><b>Data type:</b> %4<br>"
			"<br><b>Begin message length:</b> %5<br>"
			"<br><b>End message length:</b> %6<br>"
			"<br><b>Step length:</b> %7<br>"
			"<br><b>Noise message length:</b> %8<br>"
			"<br><b>Number of noise messages:</b> %9<br>"
			"<br><b>Number of noise processes:</b> %10<br>"
			"<br><b>Number of repeates:</b> %11</p>").\
		 arg(src->GetSourceFileName()).arg(src->GetNumProcessors()).arg(src->GetTestType()).\
		 arg(src->GetDataType()).arg(src->GetBeginMessageLength()).arg(src->GetEndMessageLength()).\
		 arg(src->GetStepLength()).arg(src->GetNoiseMessageLength()).arg(src->GetNoiseMessageNum()).\
		 arg(src->GetNoiseProcessors()).arg(src->GetNumRepeats());
}

NV::ErrCode CntrlrSingle::SetWindow (const int len_from, const int len_to) {
	this->DropWindow();
	
	size_t sz=static_cast<size_t>(this->source->GetNumProcessors()); sz=sz*sz*sizeof(double);
	const int step=this->source->GetStepLength();
	
	this->window_size=1u+static_cast<unsigned int>((len_to-len_from)/step);
	this->window=static_cast<double**>(calloc(this->window_size,sizeof(double*)));
	if (this->window==NULL)
	{
		this->window_size=0u;
		return NV::NoMem;
	}

	emit Progress(0);
	
	double **wnd=this->window,**wnd_end=this->window+this->window_size;
	int i=1;
	NV::ErrCode read_err=NV::Success;
	
	this->window_borders[0]=len_from;
	this->window_borders[1]=len_to;
	this->source->Begin(IData::Matrix,len_from);
	for ( ; wnd!=wnd_end; ++wnd,++i)
	{
		*wnd=static_cast<double*>(malloc(sz));
		if (*wnd==NULL) break;
		read_err=this->source->GetDataAndMove(*wnd);
		if (read_err!=NV::Success) break;
		emit Progress(i*100/this->window_size);
	}
	
	emit Progress(-1);
	
	if (wnd==wnd_end) return NV::Success;
	
	--i;
	this->window_size=static_cast<unsigned int>(i);
	this->window_borders[1]=len_from+i*step;
	return ((this->window_size!=0u)? NV::WndLoadedPartly : ((read_err==NV::Success)? NV::NoMem : read_err));
}

void CntrlrSingle::GetMatrixRaster (const int mes_len, MatrixRaster* &res1, MatrixRaster* &res2) {
	const int num_proc=source->GetNumProcessors();
	double *matr=static_cast<double*>(malloc(static_cast<size_t>(num_proc*num_proc)*sizeof(double)));
	
	res1=res2=NULL;
	if (matr==NULL) return;
	this->source->GetMatrix(mes_len,matr);
	try {
		res1=new MatrixRaster(matr,num_proc,num_proc);
	}
	catch (...)
	{
		free(matr);
		res1=NULL;
	}
}

void CntrlrSingle::GetRowRaster (const int row, MatrixRaster* &res1, MatrixRaster* &res2) const {
	const unsigned int cols=static_cast<const unsigned int>(source->GetNumProcessors());
	const size_t add_matr_off=cols*sizeof(double);
	double *add_matr=static_cast<double*>(malloc(this->window_size*add_matr_off));
	const unsigned int offset=cols*static_cast<const unsigned int>(row);
	double *matr=add_matr;
	
	res1=res2=NULL;
	for (double **wnd=this->window,**wnd_end=this->window+this->window_size; wnd!=wnd_end; ++wnd)
	{
		memcpy(matr,(*wnd)+offset,add_matr_off);
		matr+=cols;
	}
	try {
		res1=new MatrixRaster(add_matr,this->window_size,cols);
	}
	catch (...)
	{
		free(add_matr);
		res1=NULL;
	}
}

void CntrlrSingle::GetColRaster (const int col, MatrixRaster* &res1, MatrixRaster* &res2) const {
	const int cols=this->source->GetNumProcessors();
	double *add_matr=static_cast<double*>(malloc(this->window_size*cols*sizeof(double)));
	double *matr;
	int i=0,j;
	
	res1=res2=NULL;
	for (double **wnd=this->window,**wnd_end=this->window+this->window_size; wnd!=wnd_end; ++wnd)
	{
		matr=(*wnd)+col;
		for (j=i+cols; i<j; ++i,matr+=cols) // matr->rows == matr->cols
			add_matr[i]=*matr;
	}
	try {
		res1=new MatrixRaster(add_matr,this->window_size,cols);
	}
	catch (...)
	{
		free(add_matr);
		res1=NULL;
	}
}

void CntrlrSingle::GetPairRaster (const int row, const int col, double* &x_points, double* &y_points, 
								  double* &y_points_aux, unsigned int &num_points) const {
	num_points=this->window_size;
	x_points=static_cast<double*>(malloc(num_points*sizeof(double)));
	if (x_points==NULL) return;
	y_points=static_cast<double*>(malloc(num_points*sizeof(double)));
	if (y_points==NULL) { free(x_points); return; }
	y_points_aux=NULL;

	const int step=this->source->GetStepLength();
	const int offset=row*this->source->GetNumProcessors()+col;
	int p_x=this->window_borders[0];
	for (unsigned int i=0u; i<this->window_size; ++i,p_x+=step)
	{
		x_points[i]=p_x;
		y_points[i]=this->window[i][offset];
	}
}

QwtColorMap* CntrlrSingle::AllocMainCMap (const double from, const double to) const {
	QwtLinearColorMap *cmap=new QwtLinearColorMap(Qt::white,Qt::black);
	cmap->addColorStop(from,Qt::white);
	cmap->addColorStop(to,Qt::black);
	return static_cast<QwtColorMap*>(cmap);
}

