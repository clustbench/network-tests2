#include "tabviewer.h"
#include "pairviewer.h"
#include "err_msgs.h"

const QString TabViewer::my_sign("2D");

bool TabViewer::Init (void) {
	connect(this,SIGNAL(SendMessToLog(const MainWindow::MsgType,const QString&,const QString&)),parent(),
			SLOT(AddMsgToLog(const MainWindow::MsgType,const QString&,const QString&)));
	
	try {
		ui=new Ui::ui_TabViewer;
	}
	catch (...) {
		ui=NULL;
		emit SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(NV::NoMem));
		return false;
	}
	try {
		ui->setupUi(this);
	}
	catch (...) {
		delete ui;
		ui=NULL;
		emit SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(NV::NoMem));
		return false;
	}
	
	ui->progressBar->hide();

	ui->GB_showForWin->setEnabled(false); // not disable()!
	ui->B_dropWindow->setEnabled(false); // not disable()!
	
	ui->SB_row->setRange(0,controller->GetNumProcessors()-1);
	ui->SB_column->setRange(0,controller->GetNumProcessors()-1);

	ui->TE_InfoData->setTextInteractionFlags(Qt::NoTextInteraction); // deny all interactions
	ui->LW_InfoHosts->setTextInteractionFlags(Qt::NoTextInteraction); // deny all interactions

	connect(controller,SIGNAL(Progress(const int)),this,SLOT(SetProgressBarValue(const int)));

	connect(ui->B_Refresh,SIGNAL(clicked()),this,SLOT(Initialize()));
	connect(ui->SB_MatrixNumber,SIGNAL(valueChanged(double)),ui->S_WindowNumber,SLOT(setValue(double)));
	connect(ui->SB_LoadWinFrom,SIGNAL(valueChanged(double)),this,SLOT(ChangeLoadWindowBtn(void)));
	connect(ui->SB_LoadWinTo,SIGNAL(valueChanged(double)),this,SLOT(ChangeLoadWindowBtn(void)));
	connect(ui->S_WindowNumber,SIGNAL(valueChanged(double)),this,SLOT(ChangeMatrNumber(const double)));
	connect(ui->B_LoadWindow,SIGNAL(clicked()),this,SLOT(LoadWindow()));
	connect(ui->B_dropWindow,SIGNAL(clicked()),this,SLOT(DropWindow()));
	connect(ui->B_ShowMatrix,SIGNAL(clicked()),this,SLOT(ShowMesLen()));
	connect(ui->B_ShowRow,SIGNAL(clicked()),this,SLOT(ShowRow()));
	connect(ui->B_ShowColumn,SIGNAL(clicked()),this,SLOT(ShowCol()));
	connect(ui->B_ShowPair,SIGNAL(clicked()),this,SLOT(ShowPair()));
	connect(ui->mdiArea,SIGNAL(subWindowActivated(QMdiSubWindow*)),this,SLOT(SubActivated(QMdiSubWindow*)));

	Initialize();

	return true;
}

void TabViewer::Initialize () {
	QString str;
	
	controller->GetInfo(str); // information about input file(s)
	ui->TE_InfoData->setText(str);
	
	controller->GetHosts(str); // hosts' names
	ui->LW_InfoHosts->setText(str);

	const int tmp_from=controller->GetBeginMessageLength(),
			  tmp_to=controller->GetRealEndMessageLength(),
			  tmp_step=controller->GetStepLength();

	ui->S_WindowNumber->setRange(tmp_from,tmp_to,tmp_step);
	ui->SB_MatrixNumber->setRange(tmp_from,tmp_to,tmp_step);
	ui->SB_LoadWinFrom->setRange(tmp_from,tmp_to-tmp_step,tmp_step);
	ui->SB_LoadWinTo->setRange(tmp_from+tmp_step,tmp_to,tmp_step);
}

void TabViewer::LoadWindow () {
	const int tmp_from=ui->SB_LoadWinFrom->value();
	const int tmp_to=ui->SB_LoadWinTo->value();
	
	// 'tmp_from' is always less that 'tmp_to'
	
	const NV::ErrCode err=controller->SetWindow(tmp_from,tmp_to);
		
	if (!((err==NV::Success) || (err==NV::WndLoadedPartly)))
	{
		emit SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(err));
		return;
	}
	
	const int *borders=controller->GetWindowBorders();
	
	QString status(controller->GetSourceFileName());
	status+=": ";
	if (err==NV::WndLoadedPartly)
	{
		status+=tr("only part of the window with lengths from %1 to %2 was loaded").arg(borders[0]).arg(borders[1]);
		emit SendMessToLog(MainWindow::Info,my_sign,status);
		
		(status=tr("partly "))+=tr("loaded");
		ui->L_StateWinStatus->setText(status);
	}
	else
	{
		status+=tr("window with lengths from %1 to %2 was loaded").arg(borders[0]).arg(borders[1]);
		emit SendMessToLog(MainWindow::Success,my_sign,status);
		
		ui->L_StateWinStatus->setText(tr("loaded"));
	}
	
	ui->L_StateWinFrom->setText(QString::number(borders[0]));
	ui->L_StateWinTo->setText(QString::number(borders[1]));
	ui->GB_showForWin->setEnabled(true); // not enable()!
	ui->B_dropWindow->setEnabled(true); // not enable()!
	//ui->B_ShowColumn->enable();
	//ui->B_ShowPair->enable();
	//ui->B_ShowRow->enable();
}

void TabViewer::DropWindow () {
	controller->DropWindow(); // always successfull
	
	QString status(controller->GetSourceFileName());
	status+=tr(": window was dropped");
	emit SendMessToLog(MainWindow::Success,my_sign,status);
	
	ui->GB_showForWin->setEnabled(false); // not disable()!
	ui->B_dropWindow->setEnabled(false); // not disable()!
	ui->L_StateWinStatus->setText(tr("not loaded"));
	ui->L_StateWinFrom->setText(tr("N/A"));
	ui->L_StateWinTo->setText(tr("N/A"));
}

void TabViewer::ShowMesLen () {
	const int mes_len=ui->SB_MatrixNumber->value();
	MatrixRaster *matr_raster[2];
	MatrixViewer *new_m_v=new MatrixViewer(ui->mdiArea,controller,-1);

	controller->GetMatrixRaster(mes_len,matr_raster[0],matr_raster[1]);

	new_m_v->Init(tr("Matrix for message length %1").arg(mes_len),matr_raster);
	new_m_v->SetLength(QPoint(mes_len,mes_len));
	new_m_v->SetPointFrom(QPoint(0,0));
	new_m_v->SetPointTo(QPoint(controller->GetNumProcessors(),controller->GetNumProcessors()));

	NewMatrix_mes(new_m_v);

	delete matr_raster[0];
	if (matr_raster[1]!=NULL) delete matr_raster[1];
}

void TabViewer::NewMatrix_mes (MatrixViewer *m_v) {
	connect(m_v,SIGNAL(Closing(QWidget*)),this,SLOT(DeleteSubWindow(QWidget*)));
	m_v->SetNormalizeToWin(ui->GB_showForWin->isEnabled());
	connect(ui->B_LoadWindow,SIGNAL(clicked()),m_v,SLOT(SetNormalizeToWin()));
	connect(ui->B_dropWindow,SIGNAL(clicked(bool)),m_v,SLOT(SetNormalizeToWin(const bool)));
	connect(m_v,SIGNAL(RowChng(int)),this,SLOT(SetRowValue(const int)));
	connect(m_v,SIGNAL(ColChng(int)),this,SLOT(SetColValue(const int)));
	connect(m_v,SIGNAL(GiveInvariant(const int)),m_v,SLOT(GetRowAndCol()));
	connect(m_v,SIGNAL(ZoomMatrix(MatrixViewer*)),this,SLOT(NewMatrix_mes(MatrixViewer*)));
	ui->mdiArea->addSubWindow(m_v);
	m_v->show();
}

void TabViewer::ShowRow () {
	const int row=ui->SB_row->value();
	MatrixRaster *matr_raster[2];
	const QPoint len(controller->GetWindowBorders()[0],controller->GetWindowBorders()[1]);
	MatrixViewer *new_m_v=new MatrixViewer(ui->mdiArea,controller,row);    
	controller->GetRowRaster(row,matr_raster[0],matr_raster[1]);

	new_m_v->Init(tr("Row %1, message lengths from %2 to %3").arg(row).arg(len.x()).arg(len.y()),matr_raster);
	new_m_v->SetLength(len);
	new_m_v->SetPointFrom(QPoint(0,0));
	new_m_v->SetPointTo(QPoint(controller->GetNumProcessors(),(len.y()-len.x())/controller->GetStepLength()+1));

	NewMatrix_row(new_m_v);

	delete matr_raster[0];
	if (matr_raster[1]!=NULL) delete matr_raster[1];
}

void TabViewer::NewMatrix_row (MatrixViewer *m_v) {
	connect(m_v,SIGNAL(Closing(QWidget*)),this,SLOT(DeleteSubWindow(QWidget*)));
	m_v->SetNormalizeToWin(ui->GB_showForWin->isEnabled());
	connect(ui->B_LoadWindow,SIGNAL(clicked()),m_v,SLOT(SetNormalizeToWin()));
	connect(ui->B_dropWindow,SIGNAL(clicked(bool)),m_v,SLOT(SetNormalizeToWin(const bool)));
	connect(m_v,SIGNAL(ColChng(int)),this,SLOT(SetColValue(const int)));
	connect(m_v,SIGNAL(GiveInvariant(const int)),this,SLOT(SetRowValue(const int)));
	connect(m_v,SIGNAL(ZoomMatrix(MatrixViewer*)),this,SLOT(NewMatrix_row(MatrixViewer*)));
	ui->mdiArea->addSubWindow(m_v);
	m_v->show();
}

void TabViewer::ShowCol () {
	const int col=ui->SB_column->value();
	MatrixRaster *matr_raster[2];
	const QPoint len(controller->GetWindowBorders()[0],controller->GetWindowBorders()[1]);
	MatrixViewer *new_m_v=new MatrixViewer(ui->mdiArea,controller,col);

	controller->GetColRaster(col,matr_raster[0],matr_raster[1]);

	new_m_v->Init(tr("Column %1, message lengths from %2 to %3").arg(col).arg(len.x()).arg(len.y()),matr_raster);
	new_m_v->SetLength(len);
	new_m_v->SetPointFrom(QPoint(0,0));
	new_m_v->SetPointTo(QPoint(controller->GetNumProcessors(),(len.y()-len.x())/controller->GetStepLength()+1));

	NewMatrix_col(new_m_v);

	delete matr_raster[0];
	if (matr_raster[1]!=NULL) delete matr_raster[1];
}

void TabViewer::NewMatrix_col (MatrixViewer *m_v) {
	connect(m_v,SIGNAL(Closing(QWidget*)),this,SLOT(DeleteSubWindow(QWidget*)));
	m_v->SetNormalizeToWin(ui->GB_showForWin->isEnabled());
	connect(ui->B_LoadWindow,SIGNAL(clicked()),m_v,SLOT(SetNormalizeToWin()));
	connect(ui->B_dropWindow,SIGNAL(clicked(bool)),m_v,SLOT(SetNormalizeToWin(const bool)));
	connect(m_v,SIGNAL(ColChng(int)),this,SLOT(SetRowValue(const int)));
	connect(m_v,SIGNAL(GiveInvariant(const int)),this,SLOT(SetColValue(const int)));
	connect(m_v,SIGNAL(ZoomMatrix(MatrixViewer*)),this,SLOT(NewMatrix_col(MatrixViewer*)));
	ui->mdiArea->addSubWindow(m_v);
	m_v->show();
}

void TabViewer::ShowPair () {
	const int row=ui->SB_row->value(),col=ui->SB_column->value();
	double *x_points=NULL,*y_points=NULL,*y_points_aux=NULL;
	unsigned int num_pnts;
	PairViewer *new_pair_view;
	const QString title(tr("Point (%1,%2): message lengths from %3 to %4").arg(row).arg(col).\
						arg(controller->GetWindowBorders()[0]).arg(controller->GetWindowBorders()[1]));

	try {
		new_pair_view=new PairViewer(ui->mdiArea);
	}
	catch (...) { return; }
	connect(new_pair_view,SIGNAL(Closing(QWidget*)),this,SLOT(DeleteSubWindow(QWidget*)));
	controller->GetPairRaster(row,col,x_points,y_points,y_points_aux,num_pnts);
	new_pair_view->Init(title,x_points,y_points,y_points_aux,num_pnts,controller->GetType());
	ui->mdiArea->addSubWindow(new_pair_view);
	new_pair_view->show();
	if (x_points!=NULL) free(x_points);
	if (y_points!=NULL) free(y_points);
	if (y_points_aux!=NULL) free(y_points_aux);
}

