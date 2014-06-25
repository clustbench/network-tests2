#include "matrixviewer.h"
#include <cfloat>

void MatrixViewer::Init (const QString &title, MatrixRaster* data[2]) {
	try {
		ui=new Ui::ui_MatrixViewer;
	}
	catch (...)
	{
		ui=NULL;
		return;
	}
	try {
		ui->setupUi(this);
	}
	catch (...)
	{
		delete ui;
		ui=NULL;
		return;
	}

	this->setWindowTitle(title);
	this->setFocusPolicy(Qt::ClickFocus);

	ui->Plot->setAxisScale(QwtPlot::yLeft,data[0]->GetRows(),0.0);
	ui->Plot->setAxisScale(QwtPlot::xBottom,0.0,data[0]->GetCols());
	ui->Plot->enableAxis(QwtPlot::xTop,false);
	ui->Plot->enableAxis(QwtPlot::xBottom,false);
	ui->Plot->enableAxis(QwtPlot::yLeft,false);
	ui->Plot->enableAxis(QwtPlot::yRight,false);

	//QwtColorMap *c_map=_cntrl->AllocMainCMap(ui->S_Left->minValue(),ui->S_Right->maxValue());
	QwtColorMap *c_map=_cntrl->AllocMainCMap(0.0,1.0);

	QwtPlotSpectrogram *dt=new QwtPlotSpectrogram;
	dt->setData(*(data[0]));
	dt->attach(ui->Plot);
	dt->setColorMap(*c_map);
	_cntrl->FreeMainCMap(c_map);
	_data[0]=dt;
	dt->show();

	const QwtDoubleInterval tmp_range=_data[0]->data().range();
	ui->S_Left->setRange(tmp_range.minValue(),tmp_range.maxValue());
	ui->S_Left->setValue(ui->S_Left->minValue());
	ui->S_Right->setRange(ui->S_Left->minValue(),ui->S_Left->maxValue());
	ui->S_Right->setValue(ui->S_Right->maxValue());

	if (data[1]!=NULL)
	{
		// then _cntrl->AllocAuxCMap()!=NULL too

		dt=new QwtPlotSpectrogram;
		dt->setData(*(data[1]));
		dt->attach(ui->Plot);
		c_map=_cntrl->AllocAuxCMap();
		dt->setColorMap(*c_map);
		_cntrl->FreeAuxCMap(c_map);
		_data[1]=dt;
		dt->show();
	}
	else _data[1]=NULL;

	QPen pen;
	/* a pen for cursor and selection rectangle */
	pen.setColor((_cntrl->GetType()==2)/*comparator*/? Qt::green : Qt::cyan);
	pen.setWidth(1);

	/* cursor on the widget with matrices (looks like '+') */
	const double data_x[]={0.5,0.5,0.5,0.0,1.0};
	const double data_y[]={0.0,1.0,0.5,0.5,0.5};
	cursor=new QwtPlotCurve;
	cursor->setPen(pen);
	cursor->setData(data_x,data_y,5);
	cursor->attach(ui->Plot);
	cursor->show();

	selection_rect=new QwtPlotCurve;
	selection_rect->show();
	selection_rect->setPen(pen);
	selection_rect->attach(ui->Plot);

	zoomer=new QwtPlotPicker(ui->Plot->canvas());
	zoomer->setSelectionFlags(QwtPicker::RectSelection | QwtPicker::CornerToCorner | QwtPicker::DragSelection);
	zoomer->setRubberBand(QwtPicker::RectRubberBand);
	zoomer->setRubberBandPen(QPen(pen.color(),1,Qt::DashLine));
	zoomer->setEnabled(true);

	connect(ui->S_Left,SIGNAL(valueChanged(double)),this,SLOT(SetRightSldrMinVal(const double)));
	connect(ui->S_Right,SIGNAL(valueChanged(double)),this,SLOT(SetLeftSldrMaxVal(const double)));
	connect(ui->tB_showInfo,SIGNAL(clicked()),this,SLOT(ShowInfo()));
	connect(ui->SB_xFrom,SIGNAL(valueChanged(int)),this,SLOT(SetAim()));
	connect(ui->SB_yFrom,SIGNAL(valueChanged(int)),this,SLOT(SetAim()));

	connect(zoomer,SIGNAL(selected(const QwtDoubleRect&)),this,SLOT(RectSelected(const QwtDoubleRect&)));

	connect(ui->SB_xFrom,SIGNAL(valueChanged(int)),this,SLOT(DrawSelectionRect()));
	connect(ui->SB_yFrom,SIGNAL(valueChanged(int)),this,SLOT(DrawSelectionRect()));
	connect(ui->SB_xTo,SIGNAL(valueChanged(int)),this,SLOT(DrawSelectionRect()));
	connect(ui->SB_yTo,SIGNAL(valueChanged(int)),this,SLOT(DrawSelectionRect()));

	connect(ui->B_zoom,SIGNAL(clicked()),this,SLOT(ShowZoom()));

	connect(ui->SB_yFrom,SIGNAL(valueChanged(int)),this,SIGNAL(RowChng(int))); // 'y' means 'row'!
	connect(ui->SB_xFrom,SIGNAL(valueChanged(int)),this,SIGNAL(ColChng(int))); // 'x' means 'column'!

	ShowInfo();

	DrawSelectionRect();
}

void MatrixViewer::SetLength (const QPoint &len) {
	_length=len;
	SetInfo();
}

void MatrixViewer::SetPointFrom (const QPoint &pnt) {
	_p_from=pnt;
	ui->SB_xFrom->setMinimum(_p_from.x());
	ui->SB_yFrom->setMinimum(_p_from.y());
	ui->SB_xTo->setMinimum(_p_from.x());
	ui->SB_yTo->setMinimum(_p_from.y());
	SetInfo();
}

void MatrixViewer::SetPointTo (const QPoint &pnt) {
	_p_to.setX(pnt.x()-1);
	_p_to.setY(pnt.y()-1);
	ui->SB_xFrom->setMaximum(_p_to.x());
	ui->SB_yFrom->setMaximum(_p_to.y());
	ui->SB_xTo->setMaximum(_p_to.x());
	ui->SB_yTo->setMaximum(_p_to.y());
	SetInfo();
}

void MatrixViewer::SetInfo () {
	QString len_string;

	if (_length.x()==_length.y())
		len_string=tr("Matrix for message length %1").arg(_length.x());
	else
		len_string=tr("Matrix for window messages from %1 to %2").arg(_length.x()).arg(_length.y());

	len_string+=tr("\n\nRectangle: (%1; %2) - (%3; %4)").arg(_p_from.x()).arg(_p_from.y()).arg(_p_to.x()).\
														 arg(_p_to.y());

	ui->TB_info->setPlainText(len_string);
}

void MatrixViewer::ShowInfo () {
	switch (ui->tB_showInfo->arrowType())
	{
		case Qt::UpArrow:
			ui->TB_info->hide();
			ui->tB_showInfo->setArrowType(Qt::DownArrow);
			break;
		case Qt::DownArrow:
			ui->TB_info->show();
			ui->tB_showInfo->setArrowType(Qt::UpArrow);
			break;
		default:
			break;
	}
}

void MatrixViewer::SetAim () {
	const double x=ui->SB_xFrom->value()+0.5-static_cast<const double>(_p_from.x());
	const double y=ui->SB_yFrom->value()+0.5-static_cast<const double>(_p_from.y());
	const double data_x[]={x,x,x,x-0.5,x+0.5};
	const double data_y[]={y-0.5,y+0.5,y,y,y};

	cursor->show();
	cursor->setData(data_x,data_y,5);
	ui->Plot->replot();
}

void MatrixViewer::RectSelected (const QwtDoubleRect &rect) {
	ui->SB_xFrom->setValue((int)rect.left()+_p_from.x());
	ui->SB_yFrom->setValue((int)rect.top()+_p_from.y());
	ui->SB_xTo->setValue((int)rect.right()+_p_from.x());
	ui->SB_yTo->setValue((int)rect.bottom()+_p_from.y());
}

void MatrixViewer::DrawSelectionRect () {
	const double val1_x=ui->SB_xFrom->value()+0.5-static_cast<double>(_p_from.x());
	const double val2_x=ui->SB_xTo->value()+0.5-static_cast<double>(_p_from.x());
	const double val1_y=ui->SB_yFrom->value()+0.5-static_cast<double>(_p_from.y());
	const double val2_y=ui->SB_yTo->value()+0.5-static_cast<double>(_p_from.y());
	const bool same_val_x=((val1_x<val2_x+0.001) && (val2_x<val1_x+0.001));
	const bool same_val_y=((val1_y<val2_y+0.001) && (val2_y<val1_y+0.001));

	ui->B_zoom->setDisabled(same_val_x && same_val_y);

	const MatrixRaster &da=static_cast<const MatrixRaster&>(_data[0]->data());
	ui->LE_valFrom->setText(QString::number(da.value(val1_x-0.5,val1_y-0.5)));
	ui->LE_valTo->setText(QString::number(da.value(val2_x-0.5,val2_y-0.5)));

	if (same_val_x || same_val_y)
		// selection rectangle collapsed to single line or point
		SetAim(); // draw the cursor
	else cursor->hide();

	const double x_rect[]={val1_x,val2_x,val2_x,val1_x,val1_x};
	const double y_rect[]={val1_y,val1_y,val2_y,val2_y,val1_y};
	selection_rect->setData(x_rect,y_rect,5);
		
	ui->Plot->replot();
}

void MatrixViewer::SetRightSldrMinVal (const double val) {
	ui->S_Right->setRange(val,ui->S_Right->maxValue());

	const double left_min=ui->S_Left->minValue();
	const double tmp_range=ui->S_Right->maxValue()-left_min;

	if (tmp_range<DBL_EPSILON) return;

	QwtColorMap *c_map=_cntrl->AllocMainCMap((ui->S_Left->value()-left_min)/tmp_range,
											 (ui->S_Right->value()-left_min)/tmp_range);
	_data[0]->setColorMap(*c_map);
	_cntrl->FreeMainCMap(c_map);

	//connect(this, SIGNAL(SetNormalizeToWinActive(bool)), ui->RB_normalizeCurrWindow, SLOT(setEnabled(bool)));
	// the line above is incorrect, may be this line is correct:
	//connect(ui->RB_normalizeCurrWindow,SIGNAL(clicked(bool)),this,SLOT(SetNormalizeToWin(const bool)));

	ui->Plot->replot();
}

void MatrixViewer::SetLeftSldrMaxVal (const double val) {
	const double left_min=ui->S_Left->minValue();

	ui->S_Left->setRange(left_min,val);

	const double tmp_range=ui->S_Right->maxValue()-left_min;

	if (tmp_range<DBL_EPSILON) return;

	QwtColorMap *c_map=_cntrl->AllocMainCMap((ui->S_Left->value()-left_min)/tmp_range,
											 (ui->S_Right->value()-left_min)/tmp_range);
	_data[0]->setColorMap(*c_map);
	_cntrl->FreeMainCMap(c_map);

	ui->Plot->replot();
}

void MatrixViewer::ShowZoom () {
	MatrixRaster* tmp_m_r_list[2];
	double *tmp_mtr,*tmp_mtr_cur;
	const double *tmp_m_r_data;
	const MatrixRaster *tmp_m_r;
	const int from_x=ui->SB_xFrom->value(),from_y=ui->SB_yFrom->value();
	const int to_x=ui->SB_xTo->value()+1,to_y=ui->SB_yTo->value()+1;
	const int r_low=from_y-_p_from.y(),r_high=to_y-from_y;
	const int c_low=from_x-_p_from.x(),c_high=to_x-from_x;
	int cols,r;

	tmp_m_r=static_cast<const MatrixRaster*>(&(_data[0]->data()));
	cols=tmp_m_r->GetCols();
	tmp_mtr=static_cast<double*>(malloc(r_high*c_high*sizeof(double)));
	if (tmp_mtr==NULL) return;
	tmp_m_r_data=tmp_m_r->Data()+(r_low*cols+c_low);
	tmp_mtr_cur=tmp_mtr;
	for (r=0; r<r_high; ++r,tmp_m_r_data+=cols,tmp_mtr_cur+=c_high)
		memcpy(tmp_mtr_cur,tmp_m_r_data,c_high*sizeof(double));
	tmp_m_r_list[0]=new MatrixRaster(tmp_mtr,r_high,c_high);
	if (_data[1]!=NULL)
	{
		tmp_m_r=static_cast<const MatrixRaster*>(&(_data[1]->data()));
		cols=tmp_m_r->GetCols();
		tmp_mtr=static_cast<double*>(malloc(r_high*c_high*sizeof(double)));
		if (tmp_mtr==NULL)
			tmp_m_r_list[1]=NULL;
		else
		{
			tmp_m_r_data=tmp_m_r->Data()+(r_low*cols+c_low);
			tmp_mtr_cur=tmp_mtr;
			for (r=0; r<r_high; ++r,tmp_m_r_data+=cols,tmp_mtr_cur+=c_high)
				memcpy(tmp_mtr_cur,tmp_m_r_data,c_high*sizeof(double));
			tmp_m_r_list[1]=new MatrixRaster(tmp_mtr,r_high,c_high);
		}
	}
	else tmp_m_r_list[1]=NULL;

	const QString &title=this->windowTitle();
	const int zoomed_ind=title.lastIndexOf(tr(": zoomed"));
	QString z_title=(zoomed_ind>0)? title.left(zoomed_ind) : title;
	(z_title+=tr(": zoomed"))+=QString(" (%1,%2)-(%3,%4)").arg(from_x).arg(from_y).arg(to_x-1).arg(to_y-1);

	MatrixViewer *tmp_m_v=new MatrixViewer(static_cast<QMdiArea*>(parentWidget()),_cntrl,inv);
	tmp_m_v->Init(z_title,tmp_m_r_list);
	tmp_m_v->SetLength(_length);
	tmp_m_v->SetPointFrom(QPoint(from_x,from_y));
	tmp_m_v->SetPointTo(QPoint(to_x,to_y));

	emit ZoomMatrix(tmp_m_v);

	delete tmp_m_r_list[0];
	if (tmp_m_r_list[1]!=NULL) delete tmp_m_r_list[1];

	//ui->Plot->adjustSize();
}

MatrixViewer::~MatrixViewer () {
	delete _data[0];
	if (_data[1]!=NULL) delete _data[1];
	delete cursor;
	delete selection_rect;
	delete zoomer;
	delete ui->horizontalLayout;
	delete ui->horizontalLayout_2;
	delete ui;
}

