#include "cntrlr_deviat.h"
#include "matrixraster.h"

CntrlrDeviation::CntrlrDeviation (IData *data1, IData *data2, NV::ErrCode &err): ICntrlr(data1) {
	source2=data2;
	err=(*data1==data2)? NV::Success : NV::IncmpDatDev;
}

void CntrlrDeviation::GetInfo (QString &info) const {
	const IData *src=this->source;
	
	info=tr("<p><b>Data file:</b></p>");
	info+=tr("<p><b>File path:</b> %1<br>"
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
	
	info+="-----------------------------------------------";
	
	src=source2;
	info+=tr("<p><b>File with deviations:</b></p>");
	info+=tr("<p><b>File path:</b> %1<br>"
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

NV::ErrCode CntrlrDeviation::SetWindow (const int len_from, const int len_to) {
	this->DropWindow();
	
	size_t sz=static_cast<size_t>(this->source->GetNumProcessors()); sz=sz*sz*sizeof(double);
	const int step=this->source->GetStepLength();
	
	this->window_size=(1u+static_cast<unsigned int>((len_to-len_from)/step))<<1u; // double size
	this->window=static_cast<double**>(calloc(this->window_size,sizeof(double*)));
	if (this->window==NULL)
	{
		this->window_size=0u;
		return NV::NoMem;
	}

	emit Progress(0);
	
	double **wnd=this->window,**wnd_end=this->window+this->window_size;
	int i=0;
	NV::ErrCode read_err=NV::Success;
	
	this->window_borders[0]=len_from;
	this->window_borders[1]=len_to;
	this->source->Begin(IData::Matrix,len_from);
	source2->Begin(IData::Matrix,len_from);
	while (wnd!=wnd_end)
	{
		*wnd=static_cast<double*>(malloc(sz));
		if (*wnd==NULL) break;
		read_err=this->source->GetDataAndMove(*wnd);
		if (read_err!=NV::Success) break;
		++wnd;
		++i;
		emit Progress(i*100/this->window_size);
		// deviations are interleaved with values
		*wnd=static_cast<double*>(malloc(sz));
		if (*wnd==NULL) break;
		read_err=source2->GetDataAndMove(*wnd);
		if (read_err!=NV::Success) break;
		++wnd;
		++i;
		emit Progress(i*100/this->window_size);
	}
	
	emit Progress(-1);
	
	if (wnd==wnd_end) return NV::Success;
	
	i &= ~static_cast<int>(0x1); // make 'i' even
	this->window_size=static_cast<unsigned int>(i);
	this->window_borders[1]=len_from+(i>>1u)*step;
	return ((this->window_size!=0u)? NV::WndLoadedPartly : ((read_err==NV::Success)? NV::NoMem : read_err));
}

void CntrlrDeviation::GetMatrixRaster (const int mes_len, MatrixRaster* &res1, MatrixRaster* &res2) {
	const int rows=this->source->GetNumProcessors();
	const size_t sz=static_cast<size_t>(rows*rows)*sizeof(double);
	double *matr=static_cast<double*>(malloc(sz));
	
	res1=res2=NULL;
	if (matr==NULL) return;
	if (this->source->GetMatrix(mes_len,matr)!=NV::Success) { free(matr); return; }
	try {
		res1=new MatrixRaster(matr,rows,rows);
	}
	catch (...)
	{
		free(matr);
		res1=NULL;
		return;
	}
	matr=static_cast<double*>(malloc(sz));
	if (matr==NULL) { delete res1; res1=NULL; return; }
	if (source2->GetMatrix(mes_len,matr)!=NV::Success) { free(matr); delete res1; res1=NULL; return; }
	try {
		res2=new MatrixRaster(matr,rows,rows);
	}
	catch (...)
	{
		free(matr);
		delete res1;
		res1=NULL;
		res2=NULL;
	}
}

void CntrlrDeviation::GetRowRaster (const int row, MatrixRaster* &res1, MatrixRaster* &res2) const {
	const unsigned int rows=this->window_size>>1u;
	const unsigned int cols=static_cast<const unsigned int>(this->source->GetNumProcessors());
	const size_t add_matr_off=cols*sizeof(double);
	double *add_matr1,*add_matr2;
	const unsigned int offset=cols*static_cast<const unsigned int>(row);
	double **wnd=this->window,**wnd_end=this->window+this->window_size;
	double *matr1,*matr2;
	
	res1=res2=NULL;
	add_matr1=static_cast<double*>(malloc(rows*add_matr_off));
	if (add_matr1==NULL) return;
	add_matr2=static_cast<double*>(malloc(rows*add_matr_off));
	if (add_matr2==NULL) { free(add_matr1); return; }
	matr1=add_matr1;
	matr2=add_matr2;
	while (wnd!=wnd_end)
	{
		memcpy(matr1,*wnd+offset,add_matr_off);
		matr1+=cols;
		++wnd;
		memcpy(matr2,*wnd+offset,add_matr_off);
		matr2+=cols;
		++wnd;
	}
	try {
		res1=new MatrixRaster(add_matr1,rows,cols);
	}
	catch (...) { res1=NULL; free(add_matr2); free(add_matr1); return; }
	try {
		res2=new MatrixRaster(add_matr2,rows,cols);
	}
	catch (...) { res2=NULL; free(add_matr2); delete res1; res1=NULL; }
}

void CntrlrDeviation::GetColRaster (const int col, MatrixRaster* &res1, MatrixRaster* &res2) const {
	const unsigned int rows=this->window_size>>1u;
	const unsigned int cols=static_cast<const unsigned int>(source->GetNumProcessors());
	double *add_matr1,*add_matr2;
	double *matr,*matr2;
	unsigned int i=0u,j;
	double **wnd=this->window,**wnd_end=this->window+this->window_size;
	
	res1=res2=NULL;
	add_matr1=static_cast<double*>(malloc(rows*cols*sizeof(double)));
	if (add_matr1==NULL) return;
	add_matr2=static_cast<double*>(malloc(rows*cols*sizeof(double)));
	if (add_matr2==NULL) { free(add_matr1); return; }
	while (wnd!=wnd_end)
	{
		matr=*wnd+col;
		++wnd;
		matr2=*wnd+col;
		++wnd;
		for (j=i+cols; i<j; ++i) // matr->rows == matr->cols
		{
			add_matr1[i]=*matr;
			matr+=cols;
			add_matr2[i]=*matr2;
			matr2+=cols;
		}
	}
	try {
		res1=new MatrixRaster(add_matr1,rows,cols);
	}
	catch (...) { res1=NULL; free(add_matr2); free(add_matr1); return; }
	try {
		res2=new MatrixRaster(add_matr2,rows,cols);
	}
	catch (...) { res2=NULL; free(add_matr2); delete res1; res1=NULL; }
}

void CntrlrDeviation::GetPairRaster (const int row, const int col, double* &x_points, double* &y_points, 
									 double* &y_points_aux, unsigned int &num_points) const {
	const int step=this->source->GetStepLength();
	const int offset=row*this->source->GetNumProcessors()+col;
	int p_x=this->window_borders[0];
	double **wnd=this->window;
	
	num_points=this->window_size>>1u;
	x_points=(double*)malloc(num_points*sizeof(double));
	y_points=(double*)malloc(num_points*sizeof(double));
	y_points_aux=(double*)malloc(num_points*sizeof(double));
	if ((x_points==NULL) || (y_points==NULL) || (y_points_aux==NULL)) return;
	for (unsigned int i=0u; i<num_points; ++i,p_x+=step)
	{
		x_points[i]=p_x;
		y_points[i]=(*wnd)[offset];
		++wnd;
		y_points_aux[i]=(*wnd)[offset];
		++wnd;
	}
}

QwtColorMap* CntrlrDeviation::AllocMainCMap (const double from, const double to) const {
	QwtLinearColorMap *main_cmap=new QwtLinearColorMap(Qt::white,Qt::black);
	main_cmap->addColorStop(from,Qt::white);
	main_cmap->addColorStop(to,Qt::black);
	return static_cast<QwtColorMap*>(main_cmap);
}

QwtColorMap* CntrlrDeviation::AllocAuxCMap () const {
	return static_cast<QwtColorMap*>(new(std::nothrow) QwtAlphaColorMap(Qt::red));
}

void CntrlrDeviation::FreeAuxCMap (QwtColorMap *a_cmap) const { delete a_cmap; }

