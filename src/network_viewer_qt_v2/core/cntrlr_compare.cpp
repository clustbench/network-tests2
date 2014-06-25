#include "cntrlr_compare.h"
#include "matrixraster.h"

CntrlrComparison::CntrlrComparison (IData *data1, IData *data2, NV::ErrCode &err): ICntrlr(data1) {
	source2=data2;
	err=(*data1==data2)? NV::Success : NV::IncmpDat1Dat2;
}

void CntrlrComparison::GetInfo (QString &info) const {
	const IData *src=this->source;
	
	info=tr("<p><b>First data file:</b></p>");
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
	info+=tr("<p><b>Second data file:</b></p>");
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

NV::ErrCode CntrlrComparison::SetWindow (const int len_from, const int len_to) {
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
		// values from the second file are interleaved with values from the first file
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

void CntrlrComparison::GetMatrixRaster (const int mes_len, MatrixRaster* &res1, MatrixRaster* &res2) {
	const unsigned int rows=static_cast<unsigned int>(source->GetNumProcessors());
	const unsigned int matr_size=rows*rows;
	double *tmp_mtr,*tmp_mtr2;
	unsigned int i;
	
	res1=res2=NULL;
	tmp_mtr=static_cast<double*>(malloc(matr_size*sizeof(double)));
	if (tmp_mtr==NULL) return;
	tmp_mtr2=static_cast<double*>(malloc(matr_size*sizeof(double)));
	if (tmp_mtr2==NULL) { free(tmp_mtr); return; }
	if (this->source->GetMatrix(mes_len,tmp_mtr)!=NV::Success) { free(tmp_mtr2); free(tmp_mtr); return; }
	if (source2->GetMatrix(mes_len,tmp_mtr2)!=NV::Success) { free(tmp_mtr2); free(tmp_mtr); return; }
	for (i=0u; i<matr_size; ++i)
		tmp_mtr[i]-=tmp_mtr2[i]; // differences between values are taken into account
	free(tmp_mtr2);
	try {
		res1=new MatrixRaster(tmp_mtr,rows,rows);
	}
	catch (...)
	{
		free(tmp_mtr2);
		free(tmp_mtr);
	}
}

void CntrlrComparison::GetRowRaster (const int row, MatrixRaster* &res1, MatrixRaster* &res2) const {
	const unsigned int rows=this->window_size>>1u;
	const unsigned int cols=static_cast<unsigned int>(source->GetNumProcessors());
	double *add_matr=static_cast<double*>(malloc(rows*cols*sizeof(double)));
	const unsigned int offset=cols*static_cast<unsigned int>(row);
	double *matr,*matr2;
	unsigned int i=0u,j;
	
	res1=res2=NULL;
	if (add_matr==NULL) return;
	for (double **wnd=this->window,**wnd_end=this->window+this->window_size; wnd!=wnd_end; ++wnd)
	{
		matr=*wnd+offset;
		++wnd;
		matr2=*wnd+offset;
		for (j=0u; j<cols; ++i,++j)
			add_matr[i]=matr[j]-matr2[j]; // differences between values are taken into account
	}
	try {
		res1=new MatrixRaster(add_matr,rows,cols);
	}
	catch (...) { free(add_matr); }
}

void CntrlrComparison::GetColRaster (const int col, MatrixRaster* &res1, MatrixRaster* &res2) const {
	const unsigned int rows=this->window_size>>1u;
	const unsigned int cols=(const unsigned int)source->GetNumProcessors();
	double *add_matr=static_cast<double*>(malloc(rows*cols*sizeof(double)));
	double *matr,*matr2;
	unsigned int i=0u,j;
	
	res1=res2=NULL;
	if (add_matr==NULL) return;
	for (double **wnd=this->window,**wnd_end=this->window+this->window_size; wnd!=wnd_end; ++wnd)
	{
		matr=(*wnd)+col;
		++wnd;
		matr2=(*wnd)+col;
		for (j=i+cols; i<j; ++i) // matr->rows == matr->cols
		{
			add_matr[i]=*matr-*matr2; // differences between values are taken into account
			matr+=cols;
			matr2+=cols;
		}
	}
	try {
		res1=new MatrixRaster(add_matr,rows,cols);
	}
	catch (...) { free(add_matr); }
}

void CntrlrComparison::GetPairRaster (const int row, const int col, double* &x_points, double* &y_points, 
									  double* &y_points_aux, unsigned int &num_points) const {
	num_points=this->window_size>>1u;
	x_points=(double*)malloc(num_points*sizeof(double));
	y_points=(double*)malloc(num_points*sizeof(double));
	y_points_aux=(double*)malloc(num_points*sizeof(double));
	if ((x_points==NULL) || (y_points==NULL) || (y_points_aux==NULL)) return;
	
	const int step=this->source->GetStepLength();
	const int offset=row*this->source->GetNumProcessors()+col;
	int p_x=this->window_borders[0];
	double **wnd=this->window;
	for (unsigned int i=0u; i<num_points; ++i,p_x+=step)
	{
		x_points[i]=p_x;
		y_points[i]=(*wnd)[offset];
		++wnd;
		y_points_aux[i]=(*wnd)[offset];
		++wnd;
	}
}

QwtColorMap* CntrlrComparison::AllocMainCMap (const double from, const double to) const {
	QwtLinearColorMap *cmap;
	
	try {
		cmap=new QwtLinearColorMap(Qt::blue,Qt::red);
	}
	catch (...) { return NULL; }
	cmap->addColorStop(from,Qt::blue);
	cmap->addColorStop(to,Qt::red);
	cmap->addColorStop(from+(to-from)*0.5,Qt::white);
	return static_cast<QwtColorMap*>(cmap);
}

