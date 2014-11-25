#include "fullviewer.h"
#include "../core/data_netcdf.h"
#include "../core/data_text.h"
#include "../core/opencl_defs.h"
#ifdef _OPENCL // defined in "../core/opencl_defs.h"!
  #include "../core/renderer_OpenCL.h"
#endif
#include "../core/renderer_OpenMP.h"
#include <cfloat>
#ifndef Q_WS_WIN
  #include <ctime>
#else
  #include "windows.h"
#endif
#include <QProgressBar>
#include <QTextCursor>
#include <QStyle>
#include <QLabel>
#include "err_msgs.h"

#define SEND_ERR_MSG(err) emit SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(NV::err))
#define SEND_ERR_MSG1(err,arg) emit SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(NV::err,1,&arg))

#define NOT_ENOUGH_MEMORY \
		SEND_ERR_MSG(NoMem); \
		was_error=true; \
		return

const QString FullViewer::my_sign("3D");

FullViewer::FullViewer (QWidget *parent, const int mode, bool &was_error): QWidget(parent), working_mode(mode) {
	connect(this,SIGNAL(SendMessToLog(const MainWindow::MsgType,const QString&,const QString&)),parent,
			SLOT(AddMsgToLog(const MainWindow::MsgType,const QString&,const QString&)));
	connect(this,SIGNAL(TitleChanged(QWidget*,const QString&)),parent,SLOT(ChangeTabTitle(QWidget*,const QString&)));
	connect(this,SIGNAL(CloseOnEscape(QWidget*)),parent,SLOT(CloseTab(QWidget*)));

	v_file=NULL;
	d_file=NULL;
	image=NULL;
	painter=NULL;
	clr_matrix=NULL;
	pixels=NULL;
	img_w=640u;
	img_h=480u;
	draw_box_btn=NULL;
	controls_btn=NULL;
	hosts=NULL;
	info_wdg=NULL;
	info=NULL;
	dev_info=NULL;
	renderer=NULL;
	opts=NULL;
	oxy_rot=NULL;
	pt_info=NULL;
	hosts_info=NULL;
	first_render=true;
	
	if (mode>2)
	{
		SEND_ERR_MSG(UnknownMode);
		was_error=true;
		return;
	}
	
	/* creation of the renderer from most exacting one to least exacting (due to errors) */
#ifdef _OPENCL // defined in "../core/opencl_defs.h"!
	renderer=new(std::nothrow) RendererOCL(sel_cube_x,sel_cube_y,sel_cube_vis);
	if (renderer==NULL) { NOT_ENOUGH_MEMORY; }
	const QString &msg=renderer->GetError();
	if (!msg.isEmpty())
	{
		QString msg2=msg;
		msg2.replace('\n',"<br>");
		msg2.prepend("<pre>");
		msg2+="</pre>";
		delete renderer;
		renderer=NULL;
		SEND_ERR_MSG1(NoRenderer,msg2);
		if (QMessageBox::critical(this,tr("Error"),tr("<p align=\"center\">Cannot create OpenCL-based renderer.<br>"
								  "Try OpenMP-based one?</p>"),QMessageBox::Yes | QMessageBox::No)==QMessageBox::No)
		{
			was_error=true;
			return;
		}
#endif
		renderer=new(std::nothrow) RendererOMP(sel_cube_x,sel_cube_y,sel_cube_vis);
		if (renderer==NULL) { NOT_ENOUGH_MEMORY; }
		const QString &msg1=renderer->GetError();
		if (!msg1.isEmpty())
		{
			SEND_ERR_MSG1(NoRenderer,msg1);
			delete renderer;
			renderer=NULL;
			was_error=true;
			return;
		}
#ifdef _OPENCL
	}
#endif

	pixels=static_cast<unsigned int*>(malloc(img_w*img_h*sizeof(int)));
	if (pixels==NULL) { NOT_ENOUGH_MEMORY; }
	try {
		image=new QImage(reinterpret_cast<unsigned char*>(pixels),img_w,img_h,QImage::Format_RGB32);
		painter=new QPainter;
		hosts=new HostsBrowser(NULL);
		draw_box_btn=new QPushButton(NULL);
		info_wdg=new QExpandBox(NULL,IMAGE_OFFS-5u,30u);
		info=new QTextEdit(NULL);
		dev_info=new QTextEdit(NULL);
		controls_btn=new QPushButton(NULL);
		hosts_info=new QPushButton(NULL);
		opts=new RenderOpts(IMAGE_OFFS-5,mode,was_error);
	}
	catch (...)
	{ NOT_ENOUGH_MEMORY; }
	draw_box_btn->setAutoFillBackground(true);
	info_wdg->setAutoFillBackground(true);
	controls_btn->setAutoFillBackground(true);
	opts->setAutoFillBackground(true);

	/* initialize "advanced QToolBox" */
	if (info_wdg->addItem(tr("General"),info)==QExpandBox::Error) { NOT_ENOUGH_MEMORY; }
	hosts_ind=info_wdg->addItem(tr("Hosts"),hosts);
	if (hosts_ind==QExpandBox::Error) { NOT_ENOUGH_MEMORY; }
	if (info_wdg->addItem(tr("Device"),dev_info)==QExpandBox::Error) { NOT_ENOUGH_MEMORY; }
	opts_ind=info_wdg->addItem(tr("Render options"),opts);
	if (opts_ind==QExpandBox::Error) { NOT_ENOUGH_MEMORY; }
}

bool FullViewer::Init (const QString &data_filename, const QString &deviat_filename, 
					   const QString &hosts_filename, const IData::Type f_type, const QString &tab_nm) {
	QLabel progress(NULL);
	progress.setFixedSize(300,90);
	progress.move(parentWidget()->x()+((parentWidget()->width()-progress.width())>>1u),
				  parentWidget()->y()+((parentWidget()->height()-progress.height())>>1u));
	progress.setWindowModality(Qt::WindowModal);
	progress.setWindowFlags(Qt::FramelessWindowHint | Qt::X11BypassWindowManagerHint | Qt::WindowStaysOnTopHint);
	progress.setStyleSheet("background-color: rgb(255,255,240)");
	progress.setText(tr("<p align=\"center\">Reading data file(s)...</p>"));
	progress.show();
	progress.repaint(); // immediate repaint

	/* widget with hosts' names */
	hosts->setFixedSize(IMAGE_OFFS-3,127);
	hosts->setReadOnly(true);
	hosts->setTextInteractionFlags(Qt::NoTextInteraction);
	info_wdg->addButtonChild(hosts_ind,hosts_info);
	hosts_info->setFixedSize(26,26);
	hosts_info->setIcon(style()->standardIcon(QStyle::SP_MessageBoxInformation));
	hosts_info->setStyleSheet("border: node; background: none");
	hosts_info->move(hosts->width()-35,2);
	hosts_info->hide();
	connect(hosts_info,SIGNAL(clicked()),this,SLOT(ShowHostsInfo()));
	
	NV::ErrCode err=NV::Success;
	
	if (f_type==IData::NetCDF)
		v_file=new(std::nothrow) Data_NetCDF(data_filename,hosts_filename,err);
	else
		v_file=new(std::nothrow) Data_Text(data_filename,err);
	if (v_file==NULL) err=NV::NoMem;
	if (err!=NV::Success)
	{
		SendMessToLog(MainWindow::Error,my_sign,
					  ErrMsgs::ToString(err,1,(err==NV::NoHosts)? &hosts_filename : &data_filename));
		return false;
	}
	if (working_mode!=0)
	{
		if (f_type==IData::NetCDF)
			d_file=new(std::nothrow) Data_NetCDF(deviat_filename,QString()/*!*/,err);
		else
			d_file=new(std::nothrow) Data_Text(deviat_filename,err);
		if (d_file==NULL) err=NV::NoMem;
		if (err!=NV::Success)
		{
			SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(err,1,&deviat_filename));
			return false;
		}
		if (!(*v_file==d_file))
		{
			// these two files correspond to different network tests
			SEND_ERR_MSG(IncmpDat1Dat2);
			return false;
		}
	}
	
	/* get initial 'clr_matrix' sizes */
	x_num=y_num=v_file->GetNumProcessors();
	z_num=v_file->GetZNum();
	
	/* check real end message length */
	const int real_end_mes_len=v_file->GetRealEndMessageLength()+v_file->GetStepLength();
	if (real_end_mes_len!=v_file->GetEndMessageLength())
	{
		progress.hide();
		QMessageBox::warning(this,tr("Warning"),tr("<p align=center>This test was not finished.<br>"
							 "Expected %1 instead of %2<br>as end message length value.</p>").\
							 arg(real_end_mes_len).arg(v_file->GetEndMessageLength()),QMessageBox::Ok);
		progress.show();
	}
	
	QString msg=tr(" file \"");
	(msg+=data_filename)+=tr("\" is loaded");
	emit SendMessToLog(MainWindow::Success,my_sign,msg);
	if (working_mode!=0)
	{
		msg=tr(" file \"");
		(msg+=deviat_filename)+=tr("\" is loaded");
		emit SendMessToLog(MainWindow::Success,my_sign,msg);
	}
	
	if (working_mode==2)
	{
		progress.hide();
		QMessageBox::information(this,tr("Comparison of 2 files"),
								 tr("<p align=left>If a value in the first file is greater than or<br>"
								 	"equal to corresponding value in the second<br>"
								 	"file, the color is <span style=\"color:green\">green</span>, "
								 	"otherwise is <span style=\"color:red\">red</span></p>"),QMessageBox::Ok);
		progress.show();
	}

	/* renderer initialization */
	progress.setText(tr("<p align=\"center\">Initializing render system...</p>"));
	progress.repaint(); // immediate repaint
	renderer->Init(img_w,img_h,x_num,y_num,z_num);

	progress.setText(tr("<p align=\"center\">Initializing output image...</p>"));
	progress.repaint(); // immediate repaint
	/* dark-gray filling */
	memset(pixels,150,img_w*img_h*sizeof(int));
	/* add inclined hatching */
	unsigned int ii,ii61=0u,jj,tmp_ii=0u;
	for (ii=0u; ii<img_h; ++ii)
	{
		for (jj=60u-ii61; jj<img_w; jj+=60u)
			pixels[tmp_ii+jj]=0xff767676;
		tmp_ii+=img_w;
		ii61=(ii61+1u)%61u;
	}

	QFont font1(this->QWidget::font());
	font1.setPointSize(16);
	font1.setWeight(50);

	/* "Render!" button */
	draw_box_btn->setParent(this);
	draw_box_btn->setFont(font1);
	draw_box_btn->setText(tr("Render!"));
	draw_box_btn->setFixedHeight(60);
	draw_box_btn->move(IMAGE_OFFS+((img_w-13*draw_box_btn->text().length())>>1u),
					   (img_h-draw_box_btn->height())>>1u);
	connect(draw_box_btn,SIGNAL(clicked()),this,SLOT(RenderBox()));

	/* global maximum and minimum values */
	min_val1=min_val2=DBL_MAX;
	max_val1=max_val2=0.0;

	progress.setText(tr("<p align=\"center\">Filling information widgets...</p>"));
	progress.repaint(); // immediate repaint
	
	font1.setPointSize(11);
	
	info_wdg->setParent(this);
	info_wdg->setFont(font1);
	
	/* hosts' names */
	const QVector<QString> &hosts_names=v_file->GetHostNamesAsVector();
	if (hosts_names.isEmpty()) info_wdg->setItemEnabled(hosts_ind,false);
	else
	{
		// need to paint hosts' numbers in green
		int ind_of_pr; // index of parenthesis in 'hosts_names[i]'
		QString name; // resulting host's name
		
		for (QVector<QString>::const_iterator it=hosts_names.constBegin(),it_end=hosts_names.constEnd();
			 it!=it_end; ++it)
		{
			const QString &nm=*it;
			ind_of_pr=nm.indexOf(')')+1;
			try {
				((name="<span style=\"color:green\">")+=nm.left(ind_of_pr))+="</span>";
				name+=nm.right(nm.length()-ind_of_pr);
				hosts->append(name);
			}
			catch (...)
			{
				SEND_ERR_MSG(NoMem);
				return false;
			}
		}
				
		hosts->moveCursor(QTextCursor::Start);
		
		hosts_info->show();
		hosts_info->raise();
	}
	hosts->setParent(this);

	/* general info browser */
	info->setParent(this);
	info->setFixedSize(IMAGE_OFFS-3,128);
	info->setText(tr("<span style=\"color:green\">Number of processors:</span> %1<br><br>"
					 "<span style=\"color:green\">Test type:</span> %2<br><br>"
					 "<span style=\"color:green\">Data type:</span> %3<br><br>"
					 "<span style=\"color:green\">Initial message length:</span> %4<br><br>"
					 "<span style=\"color:green\">Final message length:</span> %5<br><br>"
					 "<span style=\"color:green\">Step of length:</span> %6<br><br>"
					 "<span style=\"color:green\">Noise message length:</span> %7<br><br>"
					 "<span style=\"color:green\">Number of noise messages:</span> %8<br><br>"
					 "<span style=\"color:green\">Number of noise processors:</span> %9<br><br>"
					 "<span style=\"color:green\">Number of repeats:</span> %10").\
				  arg(v_file->GetNumProcessors()).arg(v_file->GetTestType()).arg(v_file->GetDataType()).\
				  arg(v_file->GetBeginMessageLength()).arg(v_file->GetEndMessageLength()).\
				  arg(v_file->GetStepLength()).arg(v_file->GetNoiseMessageLength()).\
				  arg(v_file->GetNoiseMessageNum()).arg(v_file->GetNoiseProcessors()).arg(v_file->GetNumRepeats()));
	info->setTextInteractionFlags(Qt::NoTextInteraction);

	controls_btn->setParent(this);
	controls_btn->move(10,460);
	controls_btn->setFixedHeight(20);
	controls_btn->setFont(font1);
	controls_btn->setText(tr("Controls..."));
	connect(controls_btn,SIGNAL(clicked()),this,SLOT(ShowControls()));

	/* device info browser */
	dev_info->setParent(this);
	dev_info->setFixedSize(IMAGE_OFFS-5,30);
	dev_info->setText(tr("<span style=\"color:green\">Computing units:</span> %1").arg(renderer->NumberCUs()));
	dev_info->setTextInteractionFlags(Qt::NoTextInteraction);

	progress.setText(tr("<p align=\"center\">Generating render options...</p>"));
	progress.repaint(); // immediate repaint

	/* render options */
	opts->Init(this);

	controls_btn->raise(); // otherwise 'opts' can overlap the button

	tab_name=tab_nm;

	volume_mode=false;

	mouse_pressed=false;
	click_disabled=false;

	pt_selection=false;
	pt_selected=false;
	hst_selected=false;

	return true;
}

void FullViewer::RenderBox () {
#ifndef Q_WS_WIN
	static struct timespec tp_start,tp_end;
#else
	static LARGE_INTEGER tp_start,tp_end,freq;
#endif

	if (first_render)
	{
		first_render=false;

		/* we do not need "Render!" button anymore */
		draw_box_btn->setUpdatesEnabled(false);
		delete draw_box_btn;
		draw_box_btn=NULL;
		// immediate processing of all paint events and such
		QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents,20);

#ifndef Q_WS_WIN
		clock_gettime(CLOCK_MONOTONIC,&tp_start);
#else
		QueryPerfomanceCounter(&tp_start);
#endif

		clr_matrix=static_cast<unsigned short*>(calloc(x_num*y_num*z_num,sizeof(short))); // initializes the memory to zero!
		if (clr_matrix==NULL) CompressMatrix();

		FillMatrix(); // fill 'clr_matrix'
		renderer->SetClrMatrix(clr_matrix);

#ifndef Q_WS_WIN
		clock_gettime(CLOCK_MONOTONIC,&tp_end);
		// execution time is rounded to the second decimal digit
		tab_name+=tr(" | %1 s | ").\
			arg(0.01f*floorf((100.0f*(float)(tp_end.tv_sec-tp_start.tv_sec)+
				1.0e-7f*(float)(tp_end.tv_nsec-tp_start.tv_nsec))+0.5f));
#else
		QueryPerfomanceCounter(&tp_end);
		QueryPerfomanceFrequency(&freq);
		// execution time is rounded to the second decimal digit
		tab_name+=tr(" | %1 s | ").arg(0.01f*floorf((100.0f*((float)(tp_end.QuadPart-tp_start.QuadPart))/
													(float)freq.QuadPart)+0.5f));
#endif

		opts->ActivateAll();

		try {
			oxy_rot=new RotOXYButton(this,img_h);
		}
		catch (const std::bad_alloc&) {
			emit CloseOnEscape(this);
			return;
		}
		oxy_rot->show();

		setFocus(); // otherwise some widget (I think 'oxy_rot') steals keyboard focus
	}

#ifndef Q_WS_WIN
	clock_gettime(CLOCK_MONOTONIC,&tp_start);
#else
	QueryPerfomanceCounter(&tp_start);
#endif

	renderer->RenderBox(pixels);

#ifndef Q_WS_WIN
	clock_gettime(CLOCK_MONOTONIC,&tp_end);
	const double render_fps=1.0e+9/(1.0e+9*static_cast<double>(tp_end.tv_sec-tp_start.tv_sec)+\
									static_cast<double>(tp_end.tv_nsec-tp_start.tv_nsec));
#else
	QueryPerfomanceCounter(&tp_end);
	QueryPerfomanceFrequency(&freq);
	const double render_fps=static_cast<double>(freq.QuadPart)/
							static_cast<double>(tp_end.QuadPart-tp_start.QuadPart);
#endif

	QString title(tab_name);
	title+=QString::number(render_fps,'f',2); // write FPS to the title
	emit TitleChanged(this,title);

	update();
}

void FullViewer::ChangePtRepr (const PtRepr pt_rpr) {
	if (renderer->ChangeKernel(pt_rpr)==pt_rpr)
		return; // representation of "points" was not changed
	const QString &msg=renderer->GetError();
	if (!msg.isEmpty())
	{
		SEND_ERR_MSG1(RenderError,msg);
#ifdef _OPENCL // defined in "../core/opencl_defs.h"!
		delete renderer;
		renderer=NULL;
		if (QMessageBox::critical(this,tr("Error"),tr("<p align=center>An error occured in OpenCL-based renderer.<br>"
								  "Try OpenMP-based one?</p>"),QMessageBox::Yes | QMessageBox::No)==QMessageBox::No)
		{
			/* no further activity! */
			emit CloseOnEscape(this);
			return;
		}
		try {
			renderer=new RendererOMP(sel_cube_x,sel_cube_y,sel_cube_vis);
		}
		catch (const std::bad_alloc&) {
			renderer=NULL;
			SEND_ERR_MSG(NoMem);
			emit CloseOnEscape(this);
			return;
		}
		const QString &msg1=renderer->GetError();
		if (!msg1.isEmpty())
		{
			SEND_ERR_MSG1(NoRenderer,msg1);
			emit CloseOnEscape(this);
			return;
		}
		renderer->Init(img_w,img_h,x_num,y_num,z_num);
		renderer->SetClrMatrix(clr_matrix);
		renderer->ChangeKernel(pt_rpr);
		const QString &msg2=renderer->GetError();
		if (!msg2.isEmpty())
			SEND_ERR_MSG1(RenderError,msg2);
		else
			if (!first_render) RenderBox();
#endif
	}
	else
		if (!first_render) RenderBox();
}

#define PUT_G(clr) (static_cast<unsigned short>(static_cast<unsigned char>(clr)))
#define PUT_R(clr) (static_cast<unsigned short>(clr)<<8u)

void FullViewer::FillMatrix () {
	QProgressBar progr(this);
	progr.setFixedSize(200,30);
	progr.move(IMAGE_OFFS+((img_w-progr.width())>>1),(img_h-progr.height())>>1);
	progr.setRange(0,100);
	progr.setValue(0);
	progr.setTextVisible(true);
	QLabel progr_text(this);
	progr_text.setStyleSheet("color: white");
	progr_text.setAutoFillBackground(false);
	progr_text.setFixedSize(progr.width(),20);
	progr_text.move(progr.x(),progr.y()-25);
	progr_text.setTextInteractionFlags(Qt::NoTextInteraction);
	progr_text.setText(tr("<p align=\"center\">Building 3D model...</p>"));
	progr_text.show();
	progr.show();
	// immediate processing of all paint events
	QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents,5);
	
	/* color matrix is filled such way: 
	   for z-coordinate 0:
	   x-coordinates in row 0 | x-coordinates in row 1 | ... 
	   for z-coordinate 1: ...*/
	
	double *line=static_cast<double*>(malloc(x_num*sizeof(double))); // line of size 1xX_NUM

	if (line==NULL)
	{
		SEND_ERR_MSG(NoMem);
		emit CloseOnEscape(this);
		return;
	}
	
	double *l; // iterator for 'line'
	const double *const l_end=line+x_num;
	unsigned short *mtr; // iterator for 'clr_matrix'
	double val;
	double r_diff_max_min_v; // reciprocal of the difference between maximum and minimum values
	const int yz_num=y_num*z_num;
	int i=0;
	
	min_val1=min_val2=DBL_MAX;
	max_val1=max_val2=0.0;
	
	#define DRAW_PROGR (i & 0xff)==0
	
	switch (working_mode)
	{
		case 0: /* single file */
			// find min/max of all values at first
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			while (v_file->GetDataAndMove(line)==NV::Success)
			{
				for (l=line; l!=l_end; ++l)
				{
					val=*l;
					min_val1=(min_val1>val)? val : min_val1;
					max_val1=(max_val1<val)? val : max_val1;
				}
				i+=50;
				if (DRAW_PROGR) progr.setValue(i/yz_num);
			}
			r_diff_max_min_v=(max_val1<(min_val1+1.0e-305))? 0.0 : (255.0/(max_val1-min_val1));
			// now we can fill 'clr_matrix' (yes, the file will be read twice!)
			// stretch colours: [a,b] --> [0.0,1.0], then mult by 255
			mtr=clr_matrix;
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			while (v_file->GetDataAndMove(line)==NV::Success)
			{
				for (l=line; l!=l_end; ++l)
				{
					// green color
					*mtr=PUT_G(floor((*l-min_val1)*r_diff_max_min_v+0.5));
					++mtr;
				}
				i+=50;
				if (DRAW_PROGR) progr.setValue(i/yz_num);
			}
			break;
		case 1: /* values + deviations */
		{
			double *line2=static_cast<double*>(malloc(x_num*sizeof(double))); // line of size 1xX_NUM in 
																			  // the file with deviations
			double r_diff_max_min_d;
			double *l2,val2,coef;
			
			if (line2==NULL)
			{
				free(line);
				SEND_ERR_MSG(NoMem);
				emit CloseOnEscape(this);
				return;
			}
			
			// find min/max of all values at first
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			while (v_file->GetDataAndMove(line)==NV::Success) // you shouldn't merge this cycle and the cycle below
															  // because file caching may suffer a lot from it
			{
				for (l=line; l!=l_end; ++l)
				{
					val=*l;
					min_val1=(min_val1>val)? val : min_val1;
					max_val1=(max_val1<val)? val : max_val1;
				}
				i+=25;
				if (DRAW_PROGR) progr.setValue(i/yz_num);
			}
			r_diff_max_min_v=(max_val1<(min_val1+1.0e-305))? 0.0 : (255.0/(max_val1-min_val1));
			d_file->Begin(IData::Row,d_file->GetBeginMessageLength());
			while (d_file->GetDataAndMove(line)==NV::Success)
			{
				for (l=line; l!=l_end; ++l)
				{
					val=*l;
					min_val2=(min_val2>val)? val : min_val2;
					max_val2=(max_val2<val)? val : max_val2;
				}
				i+=25;
				if (DRAW_PROGR) progr.setValue(i/yz_num);
			}
			r_diff_max_min_d=(max_val2<(min_val2+1.0e-305))? 0.0 : (255.0/(max_val2-min_val2));
			// now we can fill 'clr_matrix' (yes, the file will be read twice!)
			// stretch colours: [a,b] --> [0.0,1.0], then mult by 255
			mtr=clr_matrix;
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			d_file->Begin(IData::Row,d_file->GetBeginMessageLength());
			while ((v_file->GetDataAndMove(line)==NV::Success) &&
				   (d_file->GetDataAndMove(line2)==NV::Success))
			{
				for (l=line,l2=line2; l!=l_end; ++l,++l2)
				{
					val=*l;
					val2=*l2;
					coef=((val<1.0e-100) || (fabs(val2)>=fabs(val)))? 1.0 : val2/val;
					*mtr=(PUT_G(floor((val-min_val1)*r_diff_max_min_v*(1.0-coef)+0.5)) | 
						  PUT_R(floor((val2-min_val2)*r_diff_max_min_d*coef+0.5)));
					++mtr;
				}
				i+=50;
				if (DRAW_PROGR) progr.setValue(i/yz_num);
			}
			free(line2);
			break;
		}
		case 2: /* compare 2 files */
		{
			double *line2=static_cast<double*>(malloc(x_num*sizeof(double))); // line of size 1xX_NUM in 
																			  // the second file
			double r_diff_max_min2;
			double *l2;
			
			if (line2==NULL)
			{
				free(line);
				SEND_ERR_MSG(NoMem);
				emit CloseOnEscape(this);
				return;
			}
			
			// find min/max of all values at first
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			d_file->Begin(IData::Row,d_file->GetBeginMessageLength());
			while ((v_file->GetDataAndMove(line)==NV::Success) &&
				   (d_file->GetDataAndMove(line2)==NV::Success))
			{
				for (l=line,l2=line2; l!=l_end; ++l,++l2)
				{
					val=*l-*l2;
					if (val<0.0)
					{
						val=-val; // 'fchs' instruction in FPU
						min_val2=(val<min_val2)? val : min_val2;
						max_val2=(max_val2<val)? val : max_val2;
					}
					else
					{
						min_val1=(val<min_val1)? val : min_val1;
						max_val1=(max_val1<val)? val : max_val1;
					}
				}
				i+=50;
				if (DRAW_PROGR) progr.setValue(i/yz_num);
			}
			r_diff_max_min_v=(max_val1<(min_val1+1.0e-305))? 0.0 : (255.0/(max_val1-min_val1));
			r_diff_max_min2=(max_val2<(min_val2+1.0e-305))? 0.0 : (255.0/(max_val2-min_val2));
			// now we can fill 'clr_matrix' (yes, the file will be read twice!)
			// stretch colours: [a,b] --> [0.0,1.0], then mult by 255
			mtr=clr_matrix;
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			d_file->Begin(IData::Row,d_file->GetBeginMessageLength());
			while ((v_file->GetDataAndMove(line)==NV::Success) &&
				   (d_file->GetDataAndMove(line2)==NV::Success))
			{
				for (l=line,l2=line2; l!=l_end; ++l,++l2)
				{
					val=*l-*l2;
					*mtr=(val<0.0)? PUT_R(floor(0.5-(val+min_val2)*r_diff_max_min2)) : 
									PUT_G(floor((val-min_val1)*r_diff_max_min_v+0.5));
					++mtr;
				}
				i+=50;
				if (DRAW_PROGR) progr.setValue(i/yz_num);
			}
			free(line2);
			break;
		}
	}
	free(line);
	progr.setValue(100);
}

void FullViewer::ReFillMatrix (const double new_min, const double new_max, const bool show_msg) {
	// comments to this function are in FullViewer::FillMatrix()

	QLabel progr_text(NULL);
	if (show_msg)
	{
		progr_text.setFixedSize(250,60);
		progr_text.setAutoFillBackground(true);
		progr_text.setParent(this);
		progr_text.move(IMAGE_OFFS+((img_w-progr_text.width())>>1u),(img_h-progr_text.height())>>1u);
		progr_text.setTextInteractionFlags(Qt::NoTextInteraction);
		progr_text.setText(tr("<p align=\"center\" style=\"color:rgb(100,100,100)\">Rebuilding 3D model...</p>"));
		progr_text.show();
		progr_text.setParent(NULL);
		// immediate processing of all paint events
		QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents,5);
	}

	memset(clr_matrix,0,x_num*y_num*z_num*sizeof(short)); // zero all 'clr_matrix'

	double *line=static_cast<double*>(malloc(x_num*sizeof(double))); // line of size 1xX_NUM
		
	if (line==NULL)
	{
		SEND_ERR_MSG(NoMem);
		emit CloseOnEscape(this);
		return;
	}
	
	double *l; // iterator for 'line'
	const double *const l_end=line+x_num;
	unsigned short *mtr; // iterator for 'clr_matrix'
	const double r_diff_max_min_v=(new_max<(new_min+1.0e-305))? 0.0 : (255.0/(new_max-new_min));
	double val1;
	
	switch (working_mode)
	{
		case 0: /* single file */
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			mtr=clr_matrix;
			while (v_file->GetDataAndMove(line)==NV::Success)
			{
				for (l=line; l!=l_end; ++l)
				{
					// green color
					val1=*l;
					*mtr|=(val1<=new_min)? PUT_G(0x0) : 
						  ((val1>=new_max)? PUT_G(0xff) : PUT_G(floor((val1-new_min)*r_diff_max_min_v+0.5)));
					++mtr;
				}
			}
			break;
		case 1: /* values + deviations */
		{
			const double coef=(max_val2-min_val2)/(max_val1-min_val1); // coefficient for 'new_min2' and 'new_max2'
			const double new_min2=coef*(new_min-min_val1)+min_val2; // proportional to 'new_min'
			const double new_max2=coef*(new_max-max_val1)+max_val2; // proportional to 'new_max'
			const double r_diff_max_min2=(new_max2<(new_min2+1.0e-305))? 0.0 : (255.0/(new_max2-new_min2));
			double *line2=static_cast<double*>(malloc(x_num*sizeof(double))); // line of size 1xX_NUM 
																			  // in the file with deviations
			double *l2,val2,koeff;

			if (line2==NULL)
			{
				free(line);
				SEND_ERR_MSG(NoMem);
				emit CloseOnEscape(this);
				return;
			}
			
			mtr=clr_matrix;
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			d_file->Begin(IData::Row,d_file->GetBeginMessageLength());
			while ((v_file->GetDataAndMove(line)==NV::Success) &&
				   (d_file->GetDataAndMove(line2)==NV::Success))
			{
				for (l=line,l2=line2; l!=l_end; ++l,++l2)
				{
					val1=*l;
					val2=*l2;
					koeff=((val1<1.0e-100) || (fabs(val2)>=fabs(val1)))? 1.0 : val2/val1;
					val1=(val1<=new_min)? 0.0 : floor((val1>=new_max)? (255.0*(1.0-koeff)+0.5) : 
													  ((val1-new_min)*r_diff_max_min_v*(1.0-koeff)+0.5));
					val2=(val2<=new_min2)? 0.0 : floor((val2>=new_max2)? (255.0*koeff+0.5) : 
													   ((val2-new_min2)*r_diff_max_min2*koeff+0.5));
					*mtr|=(PUT_G(val1) | PUT_R(val2));
					++mtr;
				}
			}
			free(line2);
			break;
		}
		case 2: /* compare 2 files */
		{
			const double coef=(max_val2-min_val2)/(max_val1-min_val1); // coefficient for 'new_min2' and 'new_max2'
			const double new_min2=coef*(new_min-min_val1)+min_val2; // proportional to 'new_min'
			const double new_max2=coef*(new_max-max_val1)+max_val2; // proportional to 'new_max'
			const double r_diff_max_min2=(new_max2<(new_min2+1.0e-305))? 0.0 : (255.0/(new_max2-new_min2));
			double *line2=static_cast<double*>(malloc(x_num*sizeof(double))); // line of size 1xX_NUM 
																			  // in the second file
			double *l2;

			if (line2==NULL)
			{
				free(line);
				SEND_ERR_MSG(NoMem);
				emit CloseOnEscape(this);
				return;
			}
			
			mtr=clr_matrix;
			v_file->Begin(IData::Row,v_file->GetBeginMessageLength());
			d_file->Begin(IData::Row,d_file->GetBeginMessageLength());
			while ((v_file->GetDataAndMove(line)==NV::Success) &&
				   (d_file->GetDataAndMove(line2)==NV::Success))
			{
				for (l=line,l2=line2; l!=l_end; ++l,++l2)
				{
					val1=*l-*l2;
					if (val1<0.0)
					{
						val1=-val1; // 'fchs' instruction in FPU
						*mtr|=(val1<=new_min2)? PUT_R(0x0) : 
							  ((val1>=new_max2)? PUT_R(0xff) : PUT_R(floor((val1-new_min2)*r_diff_max_min2+0.5)));
					}
					else
						*mtr|=(val1<=new_min)? PUT_G(0x0) : 
							  ((val1>=new_max)? PUT_G(0xff) : PUT_G(floor((val1-new_min)*r_diff_max_min_v+0.5)));
					++mtr;
				}
			}
			free(line2);
			break;
		}
	}
	free(line);
	
	renderer->SetClrMatrix(clr_matrix);
}

void FullViewer::AdjustMatrix (const double new_min_v, const double new_max_v, 
							   const double new_min_d, const double new_max_d) {
	const double r_diff_max_min_v=(max_val1<(min_val1+1.0e-305))? 0.0 : (255.0/(max_val1-min_val1));
	const unsigned char color_min_v=static_cast<unsigned char>(floor((new_min_v-min_val1)*r_diff_max_min_v+0.5));
	const unsigned char color_max_v=static_cast<unsigned char>(floor((new_max_v-min_val1)*r_diff_max_min_v+0.5));
	const double r_diff_max_min_d=(max_val2<(min_val2+1.0e-305))? 0.0 : (255.0/(max_val2-min_val2));
	const unsigned char color_min_d=static_cast<unsigned char>(floor((new_min_d-min_val2)*r_diff_max_min_d+0.5));
	const unsigned char color_max_d=static_cast<unsigned char>(floor((new_max_d-min_val2)*r_diff_max_min_d+0.5));

	if (renderer->BuildVolume(color_min_v,color_max_v,color_min_d,color_max_d))
	{
		RenderBox();
		return;
	}

	/* error in the renderer!! */
	const QString &msg=renderer->GetError();
	SEND_ERR_MSG1(RenderError,msg);
#ifdef _OPENCL // defined in "../core/opencl_defs.h"!
	delete renderer;
	renderer=NULL;
	if (QMessageBox::critical(this,tr("Error"),tr("<p align=center>An error occured in OpenCL-based renderer.<br>"
							  "Try OpenMP-based one?</p>"),QMessageBox::Yes | QMessageBox::No)==QMessageBox::No)
	{
		/* no further activity! */
		emit CloseOnEscape(this);
		return;
	}
	try {
		renderer=new RendererOMP(sel_cube_x,sel_cube_y,sel_cube_vis);
	}
	catch (const std::bad_alloc&) {
		SEND_ERR_MSG(NoMem);
		emit CloseOnEscape(this);
		return;
	}
	const QString &msg1=renderer->GetError();
	if (msg1.length()>0)
	{
		SEND_ERR_MSG1(NoRenderer,msg1);
		emit CloseOnEscape(this);
		return;
	}
	renderer->Init(img_w,img_h,x_num,y_num,z_num);
	renderer->SetClrMatrix(clr_matrix);
	if (renderer->BuildVolume(color_min_v,color_max_v,color_min_d,color_max_d))
		RenderBox();
	else
		SEND_ERR_MSG1(RenderError,renderer->GetError());
#endif
}  

void FullViewer::LeaveChosenInMatrix (const Coords *const pos, const int num) {
	const size_t x_n=static_cast<size_t>(x_num),y_n=static_cast<size_t>(y_num);

	if (num<2)
	{
		const size_t ch_pnt=(static_cast<size_t>(pos[0].z)*y_n+static_cast<size_t>(pos[0].y))*x_n+
							static_cast<size_t>(pos[0].x);

		memset(clr_matrix,0,ch_pnt*sizeof(short));
		memset(clr_matrix+(ch_pnt+1u),0,(x_n*y_n*static_cast<size_t>(z_num)-(ch_pnt+1u))*sizeof(short));
	}
	else
	{
		Coords *pos_sort=static_cast<Coords*>(malloc(static_cast<size_t>(num)*sizeof(Coords)));
		int i,tmp1,tmp2,tmp3;
		Coords *cur;

		if (pos_sort==NULL)
		{
			SEND_ERR_MSG(NoMem);
			emit CloseOnEscape(this);
			return;
		}

		// sort 'pos': first - by z-coordinate, then by y-coordinate and at last by x-coordinate
		pos_sort[0]=pos[0];
		for (i=1; i<num; ++i)
		{
			tmp1=pos[i].z;
			for (cur=pos_sort+i; (cur!=pos_sort) && ((cur-1)->z>tmp1); --cur)
				*cur=*(cur-1);
			tmp2=pos[i].y;
			for ( ; (cur!=pos_sort) && ((cur-1)->z==tmp1) && ((cur-1)->y>tmp2); --cur)
				*cur=*(cur-1);
			tmp3=pos[i].x;
			for ( ; (cur!=pos_sort) && ((cur-1)->z==tmp1) && ((cur-1)->y==tmp2) && ((cur-1)->x>tmp3); --cur)
				*cur=*(cur-1);
			cur->x=tmp3;
			cur->y=tmp2;
			cur->z=tmp1;
		}

		// zero all cells except those in the array 'pos_sort'
		size_t ch_pnt1=(static_cast<size_t>(pos_sort[0].z)*y_n+static_cast<size_t>(pos_sort[0].y))*x_n+
					   static_cast<size_t>(pos_sort[0].x);
		size_t ch_pnt2=0u;
		const unsigned short inc=(working_mode==0)? 0x0001 : 0x0101;

		memset(clr_matrix,0,ch_pnt1*sizeof(short));
		clr_matrix[ch_pnt1]|=inc; // intentionally increase colours to avoid zeroes:
								  // for example, '0' will become '1', and '3' will stay '3';
								  // that's why we cannot exceed the range [0;255]
		for (i=1; i<num; ++i)
		{
			ch_pnt2=(static_cast<size_t>(pos_sort[i].z)*y_n+static_cast<size_t>(pos_sort[i].y))*x_n+
					static_cast<size_t>(pos_sort[i].x);
			memset(clr_matrix+(ch_pnt1+1u),0,(ch_pnt2-ch_pnt1-1u)*sizeof(short));
			clr_matrix[ch_pnt2]|=inc;
			ch_pnt1=ch_pnt2;
		}
		memset(clr_matrix+(ch_pnt2+1u),0,(x_n*y_n*static_cast<size_t>(z_num)-(ch_pnt2+1u))*sizeof(short));
		free(pos_sort);
	}
	
	renderer->SetClrMatrix(clr_matrix);
}

void FullViewer::LeaveChosenInMatrix (const unsigned int proc_ind) {
	// zero all cells except those which have their row OR column index equal to 'proc_ind'

	const size_t x_n=static_cast<size_t>(x_num);
	const unsigned int y_n=static_cast<unsigned int>(y_num);
	const size_t pr_ind_ss=proc_ind*sizeof(short),x_n_ss=(x_n-1u)*sizeof(short);
	const unsigned int rmndr=x_n-(proc_ind+1u);
	unsigned int j;
	const unsigned short inc=(working_mode==0)? 0x0001 : 0x0101; // see in FullViewer::LeaveChosenInMatrix (const Coords *const, const int)
	unsigned short *mtr=clr_matrix;

	// You can skip all explanations below if you can imagine how 
	// 3D-matrices are stored in 1D-arrays (in row-by-row order) ;)

	if (proc_ind!=0u)
	{
		// the beginning: go to the row with index 'proc_ind' not forgeting 
		// to process cells with column indices equal to 'proc_ind' during 
		// this "walk"; then process the whole row with index 'proc_ind'
		memset(mtr,0,pr_ind_ss); // cells before column 'proc_ind'
		mtr+=proc_ind;
		*mtr|=inc; // column 'proc_ind'
		++mtr;
		for (j=1u; j!=proc_ind; ++j)
		{
			memset(mtr,0,x_n_ss); // cells between cell {r;'proc_num'} and cell {r+1;'proc_num'}
			mtr+=(x_n-1u);
			*mtr|=inc;
			++mtr;
		}
		memset(mtr,0,x_n_ss-pr_ind_ss); // cells between cell {r;'proc_ind'} and cell {r+1;0}
		mtr+=rmndr;
	}
	for (j=0u; j!=x_n; ++j,++mtr) // row 'proc_ind'
		*mtr|=inc;
	for (int k=1; k<z_num; ++k)
	{
		// repeated part: with the help of two facts - that 'clr_matrix' is contigious 
		// and that there is constant stride between rows with index 'proc_ind' in two 
		// adjacent matrices - we can use ('x_num'-1)-stride ('y_num'-2) times (compare 
		// to avg. ('y_num'/2) times when using per-matrix algorithm!)
		memset(mtr,0,pr_ind_ss);
		mtr+=proc_ind;
		*mtr|=inc;
		++mtr;
		for (j=2u; j<y_n; ++j)
		{
			memset(mtr,0,x_n_ss);
			mtr+=(x_n-1u);
			*mtr|=inc;
			++mtr;
		}
		memset(mtr,0,x_n_ss-pr_ind_ss);
		mtr+=rmndr;
		for (j=0u; j!=x_n; ++j,++mtr)
			*mtr|=inc;
	}
	if (rmndr!=0u/*(proc_ind+1u)!=x_n*/)
	{
		// the end: there are no more rows with index 'proc_num', 
		// only column with index 'proc_ind' is taken into account; 
		// so we simply go to the end of 'clr_matrix'
		memset(mtr,0,pr_ind_ss);
		mtr+=proc_ind;
		*mtr|=inc;
		++mtr;
		for (j=2u+proc_ind; j<y_n; ++j) // number of the last cases when we can use the stride ('x_num'-1)
		{
			memset(mtr,0,x_n_ss);
			mtr+=(x_n-1u);
			*mtr|=inc;
			++mtr;
		}
		memset(mtr,0,x_n_ss-pr_ind_ss);
	}

	renderer->SetClrMatrix(clr_matrix);
}

void FullViewer::LeaveChosenInMatrix (const unsigned int *const pts, const unsigned int pt_num) {
	const size_t x_n=static_cast<size_t>(x_num);

	if ((pt_num==0u) || (pts[pt_num-1u]>=x_n)) return; // these conditions should never be true

	const unsigned int y_n=static_cast<unsigned int>(y_num);
	const unsigned short inc=(working_mode==0)? 0x0001 : 0x0101; // see in FullViewer::LeaveChosenInMatrix (const Coords *const, const int)
	unsigned short *mtr=clr_matrix;
	unsigned int pti_y,pt_y,j,i;

	for (int k=0; k<z_num; ++k)
	{
		// for each message length
		pti_y=0u;
		pt_y=pts[0];
		for (j=0u; j!=y_n; ++j)
		{
			if ((pti_y!=pt_num) && (j==pt_y))
			{
				// row with index 'pts[pti_y]'
				pt_y=pts[++pti_y];
				for (i=0u; i!=x_n; ++i)
				{
					*mtr|=inc;
					++mtr;
				}
			}
			else
			{
				// other rows: 'pts[i]' here means column index
				memset(mtr,0,pts[0]*sizeof(short));
				mtr[pts[0]]|=inc;
				for (i=1u; i!=pt_num; ++i)
				{
					memset(mtr+(pts[i-1u]+1u),0,(pts[i]-(pts[i-1u]+1u))*sizeof(short));
					mtr[pts[i]]|=inc;
				}
				memset(mtr+(pts[pt_num-1u]+1u),0,(x_n-(pts[pt_num-1u]+1u))*sizeof(short));
				mtr+=x_n;
			}
		}
	}

	renderer->SetClrMatrix(clr_matrix);
}

void FullViewer::PointInfo (const Coords &pos) {
	if (pt_info==NULL)
	{
		pt_info=new(std::nothrow) QTextEdit(this);
		if (pt_info==NULL) return;
		pt_info->setFixedSize(250,110);
		pt_info->setReadOnly(true);
		pt_info->setTextInteractionFlags(Qt::LinksAccessibleByMouse); // something that differs from 
																	  // 'Qt::NoTextInteraction' and 
																	  // 'Qt::TextEditorInteraction'
		pt_info->move(IMAGE_OFFS+img_w-pt_info->width(),1+img_h-pt_info->height()); // to right bottom corner 
																					// of the image
		pt_info->raise();
	}
		
	const int mes_len=pos.z*v_file->GetStepLength()+v_file->GetBeginMessageLength();
	double val1,val2=0.0;
	QString txt(tr("<p><span style=\"color:red\">Process-sender:</span> <span style=\"color:white\">r</span>%1<br>"
				   "<span style=\"color:blue\">Process-receiver:</span> %2<br>"
				   "<span style=\"color:yellow\">Message length:</span> <span style=\"color:white\">r</span>%3</p>"
				   "<p>").arg(pos.x).arg(pos.y).arg(mes_len));
	
	if (v_file->GetSingleValue(mes_len,pos.y,pos.x,val1)!=NV::Success) return;
	if ((working_mode!=0) && (d_file->GetSingleValue(mes_len,pos.y,pos.x,val2)!=NV::Success)) return;
	
	switch (working_mode)
	{
		case 0: /* single file */
			txt+=tr("<b>Value:</b> %1<br><span style=\"color:lightgray\"><i>Second value:</i></span></p>").arg(val1);
			break;
		case 1: /* values + deviations */
			txt+=tr("<b>Value:</b> %1<br><b>Deviation:</b> %2</p>").arg(val1).arg(val2);
			break;
		case 2: /* compare 2 files */
			txt+=tr("<b>\"Left\" value:</b> %1<br><b>\"Right\" value:</b> %2</p>").arg(val1).arg(val2);
			break;
	}
	pt_info->setText(txt);
	pt_info->show();
}

FullViewer::~FullViewer () {
	if (v_file!=NULL) delete v_file;
	if (d_file!=NULL) delete d_file;
	if (clr_matrix!=NULL) free(clr_matrix);
	if (pixels!=NULL) free(pixels);
	if (image!=NULL) delete image;
	if (painter!=NULL) delete painter;
	if (renderer!=NULL) delete renderer;
	if (draw_box_btn!=NULL) delete draw_box_btn;
	if (opts!=NULL) delete opts;
	if (controls_btn!=NULL) delete controls_btn;
	if (hosts_info!=NULL) delete hosts_info;
	if (hosts!=NULL) delete hosts;
	if (info!=NULL) delete info;
	if (dev_info!=NULL) delete dev_info;
	if (info_wdg!=NULL) delete info_wdg;
	if (oxy_rot!=NULL) delete oxy_rot;
	if (pt_info!=NULL) delete pt_info;
}

