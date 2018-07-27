#include "topoviewer.h"
#include "../core/data_text.h"
#include <netcdf.h>
#include <cmath>
#include <cfloat>
#include <set>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QMessageBox>
#include "ui_topoviewer.h"
#include <GL/glu.h>
#include <QFileDialog>
#include <utility>
#include <QCoreApplication>
#include <QTimer>
#include "err_msgs.h"

const QString TopologyViewer::my_sign("Topo");

// light-gray color for widget's background
const float TVWidget::backgr_clr = 0.9f;

// "ignore" color (white)
const GLfloat TVWidget::ignore_clr = 1.0f; /* 16.0f/51.0f */


void TopoViewerOpts::keyPressEvent (QKeyEvent *e) {
	if (e->key()==Qt::Key_Escape) return; // ignore pressing 'Esc'
}

void TopoViewerOpts::ShowMaxDistHelp (void) {
	QMessageBox::information(this,tr("Maximization of distances"),
								  tr("This feature prevents two unconnected vertices\n"
								  	 "from having the same coordinates. But maximizing\n"
								  	 "distances between unconnected vertices may affect\n"
								  	 "lengths of other edges thus decreasing the accuracy.\n"
								  	 "User is given a coefficient which provides a tradeoff\n"
								  	 "between preserving lengths of edges and physical\n"
								  	 "correctness."));
}

#define SEND_ERR_MSG(err) emit SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(NV::err))
#define SEND_ERR_MSG1(err,arg) emit SendMessToLog(MainWindow::Error,my_sign,ErrMsgs::ToString(NV::err,1,&arg))

#define NOT_ENOUGH_MEM_RET(ret) SEND_ERR_MSG(NoMem); return ret

#define NOT_ENOUGH_MEM_CLOSE SEND_ERR_MSG(NoMem); emit CloseOnEscape(this); return

TVWidget::TVWidget (void) {
	x_num=z_num=0u;
	edge_counts=NULL;
	points_x=points_y=points_z=NULL;
	geom_c_z=0.0f;
	shift_x=shift_y=shift_z=0.0f;
	alpha=beta=0.0f;
	host_names=NULL;
	min_edg_count=1u;
	show_host_names=true;
	i_v_color=NULL;
	i_e_color=NULL;
	i_e_color_val=NULL;
	mouse_pressed=click_disabled=false;
    x_move=y_move=-1; // some values to calm down compilers
    save_menu_btn= new QPushButton(NULL);
    save_heigth= new QSpinBox(NULL);
    save_width= new QSpinBox(NULL);
}

TopologyViewer::TopologyViewer (QWidget *parent, const IData::Type f_type, bool &was_error): QWidget(parent) {
	connect(this,SIGNAL(SendMessToLog(const MainWindow::MsgType,const QString&,const QString&)),parent,
			SLOT(AddMsgToLog(const MainWindow::MsgType,const QString&,const QString&)));
    connect(this,SIGNAL(TitleChanged(QWidget*,const QString&)),parent,SLOT(ChangeTabTitle(QWidget*,const QString&)));
    connect(this,SIGNAL(CloseOnEscape(QWidget*)),parent,SLOT(CloseTab(QWidget*)));

    ncdf_files=NULL;
    txt_files=NULL;
  	hor_layout=NULL;
	shmem_eps=duplex_eps=1.01;
	vals_for_edgs=2u;
	usr_z_num=0u;
	m_d_impact=1.0;
	m_d_imp_tries=0u;
	hosts_undefined=true;
	switch (f_type)
    {
      case IData::NetCDF:
		  ncdf_files=static_cast<_for_NetCDF*>(malloc(sizeof(_for_NetCDF)));
		  if (ncdf_files==NULL) { NOT_ENOUGH_MEM_RET(); }
		  ncdf_files->v_file=ncdf_files->d_file=-1;
		  ncdf_files->matr_v=ncdf_files->matr_d=-1;
		  break;
      default/*IData::Txt*/:
		  txt_files=static_cast<_for_Text*>(malloc(sizeof(_for_Text)));
		  if (txt_files==NULL) { NOT_ENOUGH_MEM_RET(); }
		  txt_files->v_file=txt_files->d_file=NULL;
		  break;
    }

	was_error=false;
}

bool TopologyViewer::Init (const bool two_files, const QString &data_filename,
						   const QString &deviat_filename, const QString &hosts_fname) {
    int num_processors=-1;
    begin_message_length=-1;
    int end_message_length=-1;
    step_length=-1;
    int noise_message_length=-1;
    int noise_message_num=-1;
    int noise_processors=-1;
    int num_repeats=-1;

    int ind_of_slash=data_filename.lastIndexOf('/');
    if (ind_of_slash<0)
    {
    	ind_of_slash=data_filename.lastIndexOf('\\');
    	if (ind_of_slash<0) ind_of_slash=-1;
    }
    fname_for_tvwidget=data_filename.right(data_filename.length()-(ind_of_slash+1));

    if (txt_files==NULL)
    {
		// reading NetCDF file(s)...

		if (nc_open(data_filename.toLocal8Bit().constData(),NC_NOWRITE,&(ncdf_files->v_file))!=NC_NOERR)
		{
			ncdf_files->v_file=-1;
			SEND_ERR_MSG1(NotANetCDF,data_filename);
			return false;
		}
		if (two_files)
		{
			if (nc_open(deviat_filename.toLocal8Bit().constData(),NC_NOWRITE,&(ncdf_files->d_file))!=NC_NOERR)
			{
				ncdf_files->d_file=-1;
				SEND_ERR_MSG1(NotANetCDF,deviat_filename);
				return false;
			}
		}

		int id;

		#define FORMATCHECK(var,var_name,error) \
			if ((nc_inq_varid(ncdf_files->v_file,var_name,&id)!=NC_NOERR) || \
			    (nc_get_var1(ncdf_files->v_file,id,NULL,&var)!=NC_NOERR)) \
				emit SendMessToLog(MainWindow::Info,my_sign,ErrMsgs::ToString(NV::error,1,&data_filename));

		FORMATCHECK(num_processors,"proc_num",NoNumProc)
		FORMATCHECK(begin_message_length,"begin_mes_length",NoBegMesLen)
		FORMATCHECK(end_message_length,"end_mes_length",NoEndMesLen)
		FORMATCHECK(step_length,"step_length",NoStepLen)
		FORMATCHECK(noise_message_length,"noise_mes_length",NoNoiseMesLen)
		FORMATCHECK(noise_message_num,"num_noise_mes",NoNoiseMesNum)
		FORMATCHECK(noise_processors,"num_noise_proc",NoNoiseNumProc)
		FORMATCHECK(num_repeats,"num_repeates",NoRpts)

		int n_vars,i;
		int *varids;
		size_t len;
		int dim_id[3];

		nc_inq_nvars(ncdf_files->v_file,&n_vars);
		varids=static_cast<int*>(malloc(n_vars*sizeof(int)));
		if (varids==NULL) { NOT_ENOUGH_MEM_RET(false); }
		nc_inq_varids(ncdf_files->v_file,&n_vars,varids);
		// find first 3D variable - suppose there will be the only one
		for (i=0; i<n_vars; ++i)
		{
		    nc_inq_varndims(ncdf_files->v_file,varids[i],&id);
		    if (id==3) break;
		}
		if (i==n_vars)
		{
		    free(varids);
		    SEND_ERR_MSG1(No3DData,data_filename);
		    return false;
		}

		ncdf_files->matr_v=varids[i]; // defined!

		free(varids);
		nc_inq_vardimid(ncdf_files->v_file,ncdf_files->matr_v,dim_id);
		/* get initial sizes of 3D matrix */
		nc_inq_dimlen(ncdf_files->v_file,dim_id[0],&len);
		if (len==0u)
		{
			SEND_ERR_MSG1(No3DData,data_filename);
		    return false;
		}
		main_wdg.z_num=len;
		nc_inq_dimlen(ncdf_files->v_file,dim_id[1],&len);
		if (len==0u)
		{
			SEND_ERR_MSG1(No3DData,data_filename);
		    return false;
		}
		main_wdg.x_num=len;
		nc_inq_dimlen(ncdf_files->v_file,dim_id[2],&len);
		if (main_wdg.x_num!=len)
		{
			SEND_ERR_MSG1(DiffWdHght,data_filename);
		    return false;
		}

		/* get real end message length */
		const int right_end_message_length=static_cast<int>(main_wdg.z_num)*step_length+begin_message_length;
		if (right_end_message_length!=end_message_length)
		{
	    	QMessageBox::warning(this,tr("Warning"),tr("<p align=center>This test was not finished.<br>"
								 "Expected %1 instead of %2<br>as end message length value.</p>")\
								 .arg(right_end_message_length-step_length).arg(end_message_length),QMessageBox::Ok);
	    	//end_message_length=right_end_message_length;
		}

		if (ncdf_files->d_file!=-1)
		{
		    nc_inq_nvars(ncdf_files->d_file,&n_vars);
		    varids=static_cast<int*>(malloc(n_vars*sizeof(int)));
		    if (varids==NULL) { NOT_ENOUGH_MEM_RET(false); }
		    nc_inq_varids(ncdf_files->d_file,&n_vars,varids);
		    // find first 3D variable - suppose there will be only one
		    for (i=0; i<n_vars; ++i)
		    {
				nc_inq_varndims(ncdf_files->d_file,varids[i],&id);
				if (id==3) break;
		    }
		    if (i==n_vars)
		    {
				free(varids);
				SEND_ERR_MSG1(No3DData,deviat_filename);
				return false;
		    }

		    ncdf_files->matr_d=varids[i]; // defined!

		    free(varids);
		    nc_inq_vardimid(ncdf_files->d_file,ncdf_files->matr_d,dim_id);
		    nc_inq_dimlen(ncdf_files->d_file,dim_id[0],&len);
		    if (len!=main_wdg.z_num)
		    {
				SEND_ERR_MSG(IncmpDat1Dat2);
				return false;
		    }
		    nc_inq_dimlen(ncdf_files->d_file,dim_id[1],&len);
		    if (len!=main_wdg.x_num)
		    {
				SEND_ERR_MSG(IncmpDat1Dat2);
				return false;
	    	}
		    nc_inq_dimlen(ncdf_files->d_file,dim_id[2],&len);
		    if (len!=main_wdg.x_num)
		    {
				SEND_ERR_MSG(IncmpDat1Dat2);
				return false;
		    }
		}

		if (!hosts_fname.isEmpty())
		{
	    	// reading hosts' names...

	    	FILE *tmp_host_f=fopen(hosts_fname.toLocal8Bit().constData(),"r");
	    	if (tmp_host_f==NULL)
	    		SEND_ERR_MSG1(NoHosts,hosts_fname);
		    else
		    {
		    	Data_Text::Line l;
				char *line;
				unsigned int host_num=0u;

				main_wdg.host_names=new(std::nothrow) QString[main_wdg.x_num];
				if (main_wdg.host_names==NULL) { NOT_ENOUGH_MEM_RET(false); }
				for ( ; ; )
				{
				    if (!Data_Text::readline(tmp_host_f,l)) break;
				    if (!l.isallws())
				    {
				    	line=l.Give_mdf();
						len=strlen(line)-1u;
						if (line[len]=='\n') line[len]='\0';
						main_wdg.host_names[host_num++]=line;
				    }
				}
				fclose(tmp_host_f);
				hosts_undefined=false;
		    }
		}
    }
    else
    {
		// reading TXT file(s)...

		txt_files->v_file=fopen(data_filename.toLocal8Bit().constData(),"r");
		if (txt_files->v_file==NULL)
		{
			SEND_ERR_MSG1(CannotOpen,data_filename);
			return false;
		}
		if (two_files)
		{
			txt_files->d_file=fopen(deviat_filename.toLocal8Bit().constData(),"r");
			if (txt_files->d_file==NULL)
			{
				SEND_ERR_MSG1(CannotOpen,data_filename);
				return false;
			}
		}

		Data_Text::Line l;
		const char *work_line;

		#define GETNEXTLINE \
			for ( ; ; ) \
			{ \
			    if (!Data_Text::readline(txt_files->v_file,l)) \
			    { \
			    	SEND_ERR_MSG1(UnexpEOF,data_filename); \
					return false; \
			    } \
			    if (!l.isallws()) break; \
			}

		#define READVAR(comment,offs,var,error) \
			work_line=strstr(l.Give(),comment); \
			if ((work_line==NULL) || (sscanf(work_line+offs,"%d",&var)<1)) \
				emit SendMessToLog(MainWindow::Info,my_sign,ErrMsgs::ToString(NV::error,1,&data_filename)); \
			else \
			{ \
			    GETNEXTLINE \
			}

		GETNEXTLINE
		work_line=strstr(l.Give(),"processors ");
		if ((work_line==NULL) || (sscanf(work_line+11,"%d",&num_processors)<1))
		{
			SEND_ERR_MSG1(NoNumProc,data_filename);
		    return false;
		}
		if (num_processors<1)
		{
			SEND_ERR_MSG1(No3DData,data_filename);
		    return false;
		}

		main_wdg.x_num=static_cast<unsigned int>(num_processors);

		for ( ; ; )
		{
		    GETNEXTLINE
		    work_line=strstr(l.Give(),"begin message length ");
		    if ((work_line!=NULL) && (sscanf(work_line+21,"%d",&begin_message_length)==1))
		    	break;
		}

		GETNEXTLINE
		READVAR("end message length ",19,end_message_length,NoEndMesLen)
		READVAR("step length ",12,step_length,NoStepLen)
		READVAR("noise message length ",21,noise_message_length,NoNoiseMesLen)
		READVAR("number of noise messages ",25,noise_message_num,NoNoiseMesNum)
		READVAR("number of noise processes ",26,noise_processors,NoNoiseNumProc)
		READVAR("number of repeates ",19,num_repeats,NoRpts)

		main_wdg.z_num=0u;
		if (strstr(l.Give(),"hosts:")!=NULL)
		{
		    // reading hosts' names...

		    unsigned int host_num=0u;
		    int last_c;

		    main_wdg.host_names=new(std::nothrow) QString[main_wdg.x_num];
		    if (main_wdg.host_names==NULL) { NOT_ENOUGH_MEM_RET(false); }
		    for ( ; ; )
		    {
				if (!Data_Text::readline(txt_files->v_file,l)) break;
				if (!l.isallws())
				{
				    if (strstr(l.Give(),"Message length ")!=NULL)
				    {
						main_wdg.z_num=1u;
						break;
				    }
				    main_wdg.host_names[host_num]=l.Give();
				    last_c=main_wdg.host_names[host_num].length()-1;
				    if (main_wdg.host_names[host_num][last_c]=='\n')
				    	main_wdg.host_names[host_num].truncate(last_c);
				    ++host_num;
				}
		    }
		    hosts_undefined=false;
		}
		else
		{
		    for ( ; ; )
		    {
				if (strstr(l.Give(),"Message length ")!=NULL)
				{
				    main_wdg.z_num=1u;
				    break;
				}
				if (!Data_Text::readline(txt_files->v_file,l)) break;
		    }
		}

		if (main_wdg.z_num==0u)
		{
			SEND_ERR_MSG1(No3DData,data_filename);
		    return false;
		}

		fgetpos(txt_files->v_file,&txt_files->matr_v_pos); // 'matr_v_pos' is set!

		/* get real end message length */
		while (Data_Text::readline(txt_files->v_file,l))
		{
		    if (strstr(l.Give(),"Message length ")!=NULL)
		    	++main_wdg.z_num;
		}
		const int wright_end_message_length=static_cast<int>(main_wdg.z_num)*step_length+begin_message_length;
		if (wright_end_message_length!=end_message_length)
		{
		    QMessageBox::warning(this,tr("Warning"),tr("<p align=center>This test was not finished.<br>"
		    					 "Expected %1 instead of %2<br>as end message length value.</p>")\
		    					 .arg(wright_end_message_length-step_length).arg(end_message_length),QMessageBox::Ok);
		    //end_message_length=wright_end_message_length;
		}

		if (txt_files->d_file!=NULL)
		{
		    int var;

		    /* check if X and Y dimensions in 'v_file' and 'd_file' are equal */
		    for ( ; ; )
		    {
				if (!Data_Text::readline(txt_files->d_file,l))
				{
					SEND_ERR_MSG1(UnexpEOF,deviat_filename);
				    return false;
				}
				if (!l.isallws()) break;
		    }
		    work_line=strstr(l.Give(),"processors ");
		    if ((work_line==NULL) || (sscanf(work_line+11,"%d",&var)<1))
		    {
		    	SEND_ERR_MSG1(NoNumProc,deviat_filename);
				return false;
		    }
		    if (var!=num_processors)
		    {
				SEND_ERR_MSG(IncmpDat1Dat2);
				return false;
		    }

		    /* check if Z dimensions in 'v_file' and 'd_file' are equal (the beginning...) */
		    var=0;
		    while (Data_Text::readline(txt_files->d_file,l))
		    {
				if (strstr(l.Give(),"Message length ")!=NULL)
				{
				    var=1;
				    break;
				}
			}
		    if (var==0)
		    {
				SEND_ERR_MSG(IncmpDat1Dat2);
				return false;
		    }

		    fgetpos(txt_files->d_file,&txt_files->matr_d_pos); // 'matr_d_pos' is set!

		    /* check if Z dimensions in 'v_file' and 'd_file' are equal (...the end) */
		    while (Data_Text::readline(txt_files->d_file,l))
		    {
				if (strstr(l.Give(),"Message length ")!=NULL) ++var;
			}
		    if (var!=static_cast<int>(main_wdg.z_num))
		    {
				SEND_ERR_MSG(IncmpDat1Dat2);
				return false;
		    }
		}

		txt_files->flt_pt=*(localeconv()->decimal_point);
    }

    QString msg=tr(" file \"");
	(msg+=data_filename)+=tr("\" is loaded");
	emit SendMessToLog(MainWindow::Success,my_sign,msg);

    if (main_wdg.host_names==NULL)
    {
		main_wdg.host_names=new(std::nothrow) QString[main_wdg.x_num];
		if (main_wdg.host_names==NULL) { NOT_ENOUGH_MEM_RET(false); }
		for (unsigned int i=0u; i!=main_wdg.x_num; ++i)
		    main_wdg.host_names[i]=QString("v%1").arg(i);
    }

    /* initializing graph matrix */
    main_wdg.edge_counts=static_cast<unsigned int*>(calloc(main_wdg.x_num*main_wdg.x_num,sizeof(int))); // fill with zeroes
    if (main_wdg.edge_counts==NULL) { NOT_ENOUGH_MEM_RET(false); }

    try { hor_layout=new QHBoxLayout(this); }
    catch (const std::bad_alloc&) { NOT_ENOUGH_MEM_RET(false); }
    hor_layout->setContentsMargins(3,3,3,3); // thin white borders
    hor_layout->addWidget(&main_wdg); // 'main_wdg' is "glued" to 'this'

    QTimer::singleShot(100,this,SLOT(Execute())); // after 100 milliseconds Execute() will be called

    return true;
}

#include <ctime>
void TopologyViewer::Execute (void) {
	/* tune TopologyViewer */
	TopoViewerOpts *options=new TopoViewerOpts(this);
	Ui::TopoOptions opts_ui;
	try {
		opts_ui.setupUi(options);
	}
	catch (...)
	{
		delete options;
		NOT_ENOUGH_MEM_CLOSE;
	}
	connect(opts_ui.maxDistHelpPB,SIGNAL(clicked()),options,SLOT(ShowMaxDistHelp()));
	opts_ui.immRedrCB->setEnabled(false);
	opts_ui.mesLenSB->setRange(begin_message_length,
							   begin_message_length+static_cast<int>(main_wdg.z_num-1u)*step_length);
	opts_ui.mesLenSB->setSingleStep(step_length);

	(void)options->exec(); // run as modal(!); it will return only QDialog::Accepted

	shmem_eps=1.0+opts_ui.shmEpsSB->value();
	duplex_eps=1.0+opts_ui.dupEpsSB->value();
	if (opts_ui.srcMesLenRB->isChecked())
	{
		vals_for_edgs=2u;
		if (step_length==0)
			usr_z_num=0u;
		else
			usr_z_num=static_cast<unsigned int>((opts_ui.mesLenSB->value()-begin_message_length)/step_length);
	}
	else
		vals_for_edgs=opts_ui.srcMedRB->isChecked()? 1u : 0u;
	unsigned int min_edg_cnt=static_cast<unsigned int>(floor(
							 static_cast<double>(main_wdg.z_num*opts_ui.nonExEdgSB->value())*0.01+0.5));
	m_d_imp_tries=0u;
	if (opts_ui.maxDistCB->isChecked())
	{
		if (opts_ui.impValManRB->isChecked())
			m_d_impact=opts_ui.impactValSB->value();
		else
		{
			m_d_impact=1.0;
			m_d_imp_tries=static_cast<unsigned int>(opts_ui.impValAutoNoTSB->value());
		}
	}
	else
		m_d_impact=0.0;
	if (opts_ui.hideEdgesCB->isChecked())
		main_wdg.min_edg_count=static_cast<unsigned int>(floor(
							   static_cast<double>(main_wdg.z_num*opts_ui.probabSB->value())*0.01+0.5));
	else main_wdg.min_edg_count=1u;
	main_wdg.show_host_names=opts_ui.showVertLblsCB->isChecked();
	delete options;
	options=NULL;

	/* go! */
	const unsigned int xy_num=main_wdg.x_num*main_wdg.x_num;
    double *matr=static_cast<double*>(malloc(xy_num*sizeof(double))); // matrix of size x_num*x_num
    if (matr==NULL) { NOT_ENOUGH_MEM_CLOSE; }

    printf("START\n\n");
    if (txt_files==NULL)
    {
		// NetCDF file

		const int file1=ncdf_files->v_file,matr1=ncdf_files->matr_v;
		size_t start[]={0u,0u,0u}; // 2nd and 3rd components are always 0
		size_t &start0=start[0];
		const size_t count[]={1u,main_wdg.x_num,main_wdg.x_num};

		for (start0=0u; start0!=main_wdg.z_num; ++start0)
		{
		    nc_get_vara_double(file1,matr1,start,count,matr);
		    if (!RetrieveTopology(matr))
			{
				free(matr);
				NOT_ENOUGH_MEM_CLOSE;
			}
		}
    }
    else
    {
		// text file

		const char float_pt=txt_files->flt_pt;
		const char non_flt_pt=(float_pt=='.')? ',' : '.';
		FILE *file1=txt_files->v_file;
		const fpos_t matr1=txt_files->matr_v_pos;
		unsigned int i,j,k;
		const unsigned int x_num1=main_wdg.x_num-1u;
		Data_Text::Line l1;
		char *line1,*splitter,*pt_pos;

		fsetpos(file1,&matr1);
		for (k=0u; k!=main_wdg.z_num; ++k)
		{
			for (j=0u; j!=main_wdg.x_num; ++j)
			{
			    Data_Text::readline(file1,l1);
			    line1=l1.Give_mdf();
			    for (i=0u; i!=x_num1; ++i)
			    {
					*(splitter=strchr(line1,'\t'))='\0';
					if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
					    *pt_pos=float_pt;
					matr[j*main_wdg.x_num+i]=atof(line1);
					line1=splitter+1;
			    }
			    if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
					*pt_pos=float_pt;
			    matr[j*main_wdg.x_num+i]=atof(line1);
			}
			if (!RetrieveTopology(matr))
			{
				free(matr);
				NOT_ENOUGH_MEM_CLOSE;
			}
		    while (Data_Text::readline(file1,l1))
		    {
		    	if (strstr(l1.Give(),"Message length ")!=NULL) break;
		    }
		}
    }
    printf("\nEND\n\n");

    /*printf("WRITE FILE\n\n");

    if (txt_files==NULL)
    {
		const int file1=ncdf_files->v_file,matr1=ncdf_files->matr_v;
		size_t start[3]={0u,0u,0u}; // 2nd and 3rd components are always 0
		const size_t count[3]={1u,main_wdg.x_num,main_wdg.x_num};

		nc_get_vara_double(file1,matr1,start,count,matr);
    }
    else
    {
		const char float_pt=txt_files->flt_pt;
		const char non_flt_pt=(float_pt=='.')? ',' : '.';
		FILE *file1=txt_files->v_file;
		const fpos_t matr1=txt_files->matr_v_pos;
		unsigned int i,j;
		const unsigned int x_num1=main_wdg.x_num-1u;
		Data_Text::Line l1;
		char *line1,*splitter,*pt_pos;

		fsetpos(file1,&matr1);
		for (j=0u; j!=main_wdg.x_num; ++j)
		{
		    Data_Text::readline(file1,l1);
		    line1=l1.Give_mdf();
		    for (i=0u; i!=x_num1; ++i)
		    {
		    	*(splitter=strchr(line1,'\t'))='\0';
		    	if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
				    *pt_pos=float_pt;
				matr[j*main_wdg.x_num+i]=atof(line1);
				line1=splitter+1;
			}
			if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
				*pt_pos=float_pt;
			matr[j*main_wdg.x_num+i]=atof(line1);
		}
    }
    FILE *ff=fopen("11.gv","w");
    fprintf(ff,"graph topology {\n");
    for (unsigned i=0u; i!=main_wdg.x_num; ++i)
    	fprintf(ff,"\tv%u [label=\"%s\"];\n",i,main_wdg.host_names[i].toLocal8Bit().constData());
    for (unsigned int i=0u; i!=main_wdg.x_num; ++i)
    {
    	for (unsigned int j=0u; j!=main_wdg.x_num; ++j)
    	{
    		if (main_wdg.edge_counts[i*main_wdg.x_num+j]>0u)
	    		fprintf(ff,"\tv%u -- v%u [label=\"%g\"];\n",i,j,0.5*(matr[i*main_wdg.x_num+j]+matr[j*main_wdg.x_num+i]));//((double)edge_counts[i*x_num+j])*100.0/(double)z_num);
    	}
    }
    fprintf(ff,"}\n");
    fclose(ff);
    ff=NULL;

    printf("WRITTEN\n\n");//return;
    */


	printf("BUILD GRAPH\n\n");

	if (!GetMatrixByValsForEdgsVar(matr))
	{
		free(matr);
		NOT_ENOUGH_MEM_CLOSE;
	}

    // forget about edges with low existence probability
    for (unsigned int *edg_c=main_wdg.edge_counts,*edg_c_end=main_wdg.edge_counts+xy_num;
    	 edg_c!=edg_c_end; ++edg_c)
    	*edg_c=(*edg_c<min_edg_cnt)? 0u : *edg_c;

    unsigned int edg_num=0u; // number of all edges
	unsigned int edg50_num=0u; // number of edges with length error not less than 50%
	unsigned int edg99_num=0u; // number of edges with length error not less than 99%

    if (!main_wdg.MapGraphInto3D(matr,m_d_impact,edg_num,edg50_num,edg99_num))
    {
    	free(matr);
		NOT_ENOUGH_MEM_CLOSE;
    }
    free(matr);
    if (edg50_num!=0u)
	{
		const double r_edg_num=100.0/static_cast<double>(edg_num);
		QString mes=tr("Graph building");
		mes+=tr(" was not precise:");
		mes+=QString("<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <b>%1%</b> ")\
						.arg(static_cast<double>(edg50_num)*r_edg_num,0,'f',2);
		mes+=tr("of edges have %1% length error or greater").arg(50); // 'arg(const)' is used to minimize
																	  // the number of translated strings
		if (edg99_num!=0u)
		{
			mes+=QString("<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <b>%1%</b> ").\
						 arg(static_cast<double>(edg99_num)*r_edg_num,0,'f',2);
			mes+=tr("of edges have %1% length error or greater").arg(99);
		}
		(mes+="<br><br>")+=tr("(the graph has <b>%1</b> edges in all)<br>").arg(edg_num);
		emit SendMessToLog(MainWindow::Info,my_sign,mes);
		QMessageBox::information(this,tr("Graph building"),mes);
	}

    main_wdg.updateGL();
}


// garbage collectors;
// make instances of these classes AFTER allocations of memory for managed pointers
// TODO: replace these 2 classes with std::unique_ptr when C++11 will be fully supported
class _FreeMeOnReturn {
	private:
		void *const ptr;

		_FreeMeOnReturn (const _FreeMeOnReturn&); // denied
		void operator= (const _FreeMeOnReturn&); // denied
	public:
		explicit _FreeMeOnReturn (void *p): ptr(p) {}
		~_FreeMeOnReturn (void) { free(ptr); }
};
template <typename T> class _Delete_arr_MeOnReturn {
	private:
		T *const arr;

		_Delete_arr_MeOnReturn (const _Delete_arr_MeOnReturn&); // denied
		void operator= (const _Delete_arr_MeOnReturn&); // denied
	public:
		explicit _Delete_arr_MeOnReturn (T *a): arr(a) {}
		~_Delete_arr_MeOnReturn (void) { delete[] arr; }
};

bool TopologyViewer::RetrieveTopology (double *matr) {
	clock_t st=clock();

	// array of vertices; each vertex stores an array of adjacent vertices
    std::vector<unsigned int> *vertices=new(std::nothrow) std::vector<unsigned int>[main_wdg.x_num];
    if (vertices==NULL) { NOT_ENOUGH_MEM_RET(false); }
    _Delete_arr_MeOnReturn<std::vector<unsigned int> > _for_vertices(vertices); // calls "delete[] vertices"
    																			// when function returns

    // distances between vertices; the structure is the same as the structure of 'vertices'
    std::vector<double> *edges=new(std::nothrow) std::vector<double>[main_wdg.x_num];
    if (edges==NULL) { NOT_ENOUGH_MEM_RET(false); }
    _Delete_arr_MeOnReturn<std::vector<double> > _for_edges(edges); // calls "delete[] edges" when function returns

    // array of flags; a flag is 'true' when corresponding vertex is "viewed" or "done"
    bool *vert_done=static_cast<bool*>(calloc(main_wdg.x_num,sizeof(bool)));
    if (vert_done==NULL) { NOT_ENOUGH_MEM_RET(false); }
    _FreeMeOnReturn _for_vert_done(vert_done); // calls "free(vert_done)" when function returns

    std::set<unsigned int> on_sh_mem; // processors on shared memory;
    								  // distances between such processors are not differ much

    unsigned int i,j=main_wdg.x_num+1u;
    unsigned int bigi=main_wdg.x_num*(main_wdg.x_num+1u); // saves a value of "<some_var>*main_wdg.x_num"
    double glob_min; // minimum value in 'matr'
    unsigned int gl_min_s,gl_min_e; // row and column index of 'glob_min' correspondently
    double *cur; // an iterator for 'matr'
    std::vector<unsigned int> *verts,*verts_end; // iterators for 'vertices'
	std::vector<double> *edgs; // an iterator for 'edges'
    bool main_break=false; // becomes 'true' when we need to exit the main cycle

    /* 1st part: retrieve spanning subgraph */

    // erase diagonal elements to avoid self-pointing
    for (i=0u; i!=bigi; i+=j)
		matr[i]=DBL_MAX;

	// start: find minimum value among values of the whole matrix;
	// it's clear that two corresponding vertices are linked
	glob_min=DBL_MAX;
	gl_min_s=0u;
	gl_min_e=0u;
	cur=matr;
	for (i=0u; i!=main_wdg.x_num; ++i)
	{
		for (j=0u; j!=main_wdg.x_num; ++j,++cur)
		{
			if (*cur<glob_min)
			{
				glob_min=*cur;
				gl_min_s=i;
				gl_min_e=j;
			}
		}
	}
	vert_done[gl_min_s]=true;
	try {
		vertices[gl_min_s].push_back(gl_min_e);
		edges[gl_min_s].push_back(glob_min);
	}
	catch (...) { NOT_ENOUGH_MEM_RET(false); }
	matr[gl_min_s*main_wdg.x_num+gl_min_e]=DBL_MAX;
	// check if edge 'gl_min_s'<->'gl_min_e' is a duplex channel (with tolerance 'shmem_eps' (!))
	glob_min*=shmem_eps;
	cur=matr+(gl_min_e*main_wdg.x_num+gl_min_s);
	if (*cur<glob_min)
	{
		try {
			vertices[gl_min_e].push_back(gl_min_s);
			edges[gl_min_e].push_back(*cur);
		}
		catch (...) { NOT_ENOUGH_MEM_RET(false); }
		*cur=DBL_MAX;
	}
	try {
		on_sh_mem.insert(gl_min_e);
	}
	catch (...) { NOT_ENOUGH_MEM_RET(false); }
	for ( ; ; )
	{
		// collect (and link) processors placed on shared memory;
		// the tolerance is 'shmem_eps'
		while (!on_sh_mem.empty())
		{
			gl_min_e=*(on_sh_mem.begin());
			on_sh_mem.erase(on_sh_mem.begin());
			cur=matr+gl_min_e*main_wdg.x_num;
			vert_done[gl_min_e]=true;
			verts=vertices+gl_min_e;
			edgs=edges+gl_min_e;
			for (j=0u; j!=main_wdg.x_num; ++j,++cur)
			{
				if (*cur<glob_min)
				{
					try {
						verts->push_back(j);
						edgs->push_back(*cur);
						if (!vert_done[j])
							on_sh_mem.insert(j);
					}
					catch (...) { NOT_ENOUGH_MEM_RET(false); }
					*cur=DBL_MAX;
				}
			}
			cur=matr+gl_min_e;
			edgs=edges;
			for (verts=vertices,verts_end=vertices+main_wdg.x_num; verts!=verts_end; ++verts,++edgs)
			{
				if (*cur<glob_min)
				{
					try {
						verts->push_back(gl_min_e);
						edgs->push_back(*cur);
					}
					catch (...) { NOT_ENOUGH_MEM_RET(false); }
					*cur=DBL_MAX;
				}
				cur+=main_wdg.x_num;
			}
		}

		/*for (i=0u; (i!=main_wdg.x_num) && vert_done[i]; ++i) ;
		if (i==main_wdg.x_num) break;*/

		for ( ; ; )
		{
			// "throw" an edge from "visited" processors to "not visited" ones;
			// this is called "ascent to cardboards";
			// find minimum value among values which correspond to
			// pairs of "visited"-"not visited" processors
			glob_min=DBL_MAX;
			gl_min_s=gl_min_e=0u;
			cur=matr;
			for (i=0u; i!=main_wdg.x_num; ++i)
			{
				if (!vert_done[i])
				{
					cur+=main_wdg.x_num;
					continue;
				}
				for (j=0u; j!=main_wdg.x_num; ++j,++cur)
				{
					if (!vert_done[j] && (*cur<glob_min))
					{
						glob_min=*cur;
						gl_min_s=i;
						gl_min_e=j;
					}
				}
			}
			if (gl_min_s==gl_min_e)
			{
				// this means that all processors are marked as "visited"
				main_break=true;
				break;
			}
			//vert_done[gl_min_s]=true;
			try {
				vertices[gl_min_s].push_back(gl_min_e);
				edges[gl_min_s].push_back(glob_min);
			}
			catch (...) { NOT_ENOUGH_MEM_RET(false); }
			matr[gl_min_s*main_wdg.x_num+gl_min_e]=DBL_MAX;
			// check if edge 'gl_min_s'<->'gl_min_e' is
			// a duplex channel (with tolerance 'duplex_eps')
			bigi=gl_min_e*main_wdg.x_num;
			cur=matr+(bigi+gl_min_s);
			if (*cur<glob_min*duplex_eps)
			{
				try {
					vertices[gl_min_e].push_back(gl_min_s);
					edges[gl_min_e].push_back(*cur);
				}
				catch (...) { NOT_ENOUGH_MEM_RET(false); }
				*cur=DBL_MAX;
			}
			// the cycle of "descent to processor cores":
			// 1) find edge with minimum length which begins in 'gl_min_e';
			//    also the length of this edge must be less than 'glob_min' (!)
			// 2) if no such edge was found, exit the cycle and go collect
			//    processors on shared memory, else go to step (3)
			// 3) 'gl_min_e' := <second end of found edge>; go to step (1)
			//
			// found edges should form a chain
			for ( ; ; )
			{
				gl_min_s=gl_min_e;
				gl_min_e=main_wdg.x_num;
				//glob_min=DBL_MAX; // new 'glob_min' must be strongly less than current one!
				cur=matr+bigi;
				for (j=0u; j!=main_wdg.x_num; ++j,++cur)
				{
					if (*cur<glob_min)
					{
						glob_min=*cur;
						gl_min_e=j;
					}
				}
				if (gl_min_e==main_wdg.x_num)
				{
					if (!vert_done[gl_min_s])
					{
						try {
							on_sh_mem.insert(gl_min_s);
						}
						catch (...) { NOT_ENOUGH_MEM_RET(false); }
					}
					break;
				}
				vert_done[gl_min_s]=true;
				try {
					vertices[gl_min_s].push_back(gl_min_e);
					edges[gl_min_s].push_back(glob_min);
				}
				catch (...) { NOT_ENOUGH_MEM_RET(false); }
				matr[bigi+gl_min_e]=DBL_MAX;
				glob_min*=shmem_eps;
				bigi=gl_min_e*main_wdg.x_num;
				cur=matr+(bigi+gl_min_s);
				if (*cur<glob_min)
				{
					try {
						vertices[gl_min_e].push_back(gl_min_s);
						edges[gl_min_e].push_back(*cur);
					}
					catch (...) { NOT_ENOUGH_MEM_RET(false); }
					*cur=DBL_MAX;
				}
			}
			if (!on_sh_mem.empty()) break;
		}
		if (main_break) break; // first part of the algorithm is done
	}

	st=clock()-st;
	printf("\n%g мс",static_cast<float>(st*1000u)/static_cast<float>(CLOCKS_PER_SEC));
	st=clock();

	/* 2nd part: retrieve "good" cycles */

	// a "good" cycle is a cycle 1->2, 2->3, ... (k-1)->k, 1->k (direction is important),
	// where |12| + |23| + ... +|k-1,k|> |1k| (|| is a length of an edge)
	std::vector<unsigned int>::const_iterator it,it_end;
	double *d=static_cast<double*>(malloc(main_wdg.x_num*sizeof(double)));

	if (d!=NULL)
	{
		_FreeMeOnReturn _for_d(d); // calls "free(d)" at the end of scope

		verts=vertices;
		edgs=edges;
		for (i=0u,bigi=0u; i!=main_wdg.x_num; ++i,bigi+=main_wdg.x_num,++verts,++edgs)
		{
			// Dijkstra algorithm for every vertex (found in Wikipedia)
			memset(vert_done,0,main_wdg.x_num*sizeof(bool));
			for (j=0u; j!=main_wdg.x_num; ++j)
				d[j]=DBL_MAX;
			d[i]=0.0;
			vert_done[i]=true;
			gl_min_e=1u; // counter of "visited" vertices
			for (it=verts->begin(),it_end=verts->end(); it!=it_end; ++it)
			{
				glob_min=(*edgs)[it-verts->begin()];
				j=*it;
				if (glob_min<d[j]) d[j]=glob_min;
			}
			for ( ; gl_min_e!=main_wdg.x_num; ++gl_min_e)
			{
				for (j=0u; vert_done[j]; ++j) ;
				glob_min=d[j];
				gl_min_s=j;
				for (++j; j!=main_wdg.x_num; ++j)
				{
					if (vert_done[j]) continue;
					if (d[j]<glob_min)
					{
						glob_min=d[j];
						gl_min_s=j;
					}
				}
				if (glob_min==DBL_MAX) break; // some vertices are unreachable
				vert_done[gl_min_s]=true;
				const std::vector<unsigned int> &v_gl_min_s=vertices[gl_min_s];
				const std::vector<double> &e_gl_min_s=edges[gl_min_s];
				for (it=v_gl_min_s.begin(),it_end=v_gl_min_s.end(); it!=it_end; ++it)
				{
					j=*it;
					if (vert_done[j]) continue;
					glob_min=d[gl_min_s]+e_gl_min_s[it-v_gl_min_s.begin()];
					if (glob_min<d[j]) d[j]=glob_min;
				}
			}
			cur=matr+bigi;
			for (j=0u; j!=main_wdg.x_num; ++j,++cur)
			{
				if ((d[j]!=DBL_MAX) && (*cur<d[j]))
				{
					//if (std::find(verts->begin(),verts->end(),j)!=verts->end()) exit(0);
					try {
						verts->push_back(j);
						edgs->push_back(*cur);
					}
					catch (...) { NOT_ENOUGH_MEM_RET(false); }
					*cur=DBL_MAX;
				}
			}
		}
	}

	st=clock()-st;
	printf(", %g мс",static_cast<float>(st*1000u)/static_cast<float>(CLOCKS_PER_SEC));
	st=clock();

	/* 3rd part: add new edges */
	unsigned int *edg_cs=main_wdg.edge_counts;
	for (verts=vertices,verts_end=vertices+main_wdg.x_num; verts!=verts_end; ++verts,edg_cs+=main_wdg.x_num)
	{
		for (it=verts->begin(),it_end=verts->end(); it!=it_end; ++it)
			++*(edg_cs+*it);
	}

	st=clock()-st;
	printf(", %g мс\n",static_cast<float>(st*1000u)/static_cast<float>(CLOCKS_PER_SEC));

	return true;
}

bool TopologyViewer::GetMatrixByValsForEdgsVar (double *matr) {
	const unsigned int xy_num=main_wdg.x_num*main_wdg.x_num;
	double val;

	if (txt_files==NULL)
    {
		const int file1=ncdf_files->v_file,matr1=ncdf_files->matr_v;
		size_t start[3]={0u,0u,0u}; // 3rd component is always equal to 0
		size_t &start0=start[0];
		size_t count[3]={1u,main_wdg.x_num,main_wdg.x_num};

		switch (vals_for_edgs)
		{
			case 0u: // simple average
			{
				nc_get_vara_double(file1,matr1,start,count,matr); // safely get the first matrix
				++start0;

				double *other_data=static_cast<double*>(malloc(xy_num*sizeof(double))); // aux. buffer
				double *iter,*o_iter;
				const double *iter_end;

				if (other_data==NULL)
				{
					// there is not enough memory to perform fast version -
					// we have to read single rows
					other_data=static_cast<double*>(malloc(main_wdg.x_num*sizeof(double)));
					if (other_data==NULL) { NOT_ENOUGH_MEM_RET(false); } // too bad
					count[1]=1u;
					size_t &start1=start[1];
					for ( ; start0!=main_wdg.z_num; ++start0)
					{
						iter=matr;
						iter_end=matr+main_wdg.x_num;
						for (start1=0u; start1!=main_wdg.x_num; ++start1)
						{
							nc_get_vara_double(file1,matr1,start,count,other_data);
							for (o_iter=other_data; iter!=iter_end; ++iter,++o_iter)
								*iter+=*o_iter;
							iter_end+=main_wdg.x_num;
						}
					}
					iter_end=matr+xy_num;
				}
				else
				{
					// fast version - read the whole matrices
					iter_end=matr+xy_num;
					for ( ; start0!=main_wdg.z_num; ++start0)
					{
						nc_get_vara_double(file1,matr1,start,count,other_data);
						for (iter=matr,o_iter=other_data; iter!=iter_end; ++iter,++o_iter)
							*iter+=*o_iter;
					}
				}
				free(other_data);
				val=1.0/static_cast<double>(main_wdg.z_num);
				for (iter=matr; iter!=iter_end; ++iter)
					*iter*=val;
				break;
			}
			case 1u: // median
			{
				const unsigned int med_z=(main_wdg.z_num-1u)>>1u;
				const unsigned int med_z_xx=med_z*main_wdg.x_num;
				double *data=static_cast<double*>(malloc(main_wdg.x_num*main_wdg.z_num*sizeof(double))); // aux
				unsigned int l,r,i,j;
				double *iter=matr,*o_iter,*o_iter1,*o_iter2;
				const double *o_iter_end;
				double tmp;

				if (data==NULL) { NOT_ENOUGH_MEM_RET(false); } // too bad
				count[1]=1u;
				count[0]=main_wdg.z_num;
				for (size_t &start1=start[1]; start1!=main_wdg.x_num; ++start1)
				{
					nc_get_vara_double(file1,matr1,start,count,data); // slow uncoalesced access!
					for (o_iter=data,o_iter_end=data+main_wdg.x_num; o_iter!=o_iter_end; ++o_iter)
					{
						/* the algorithm was taken from http://markx.narod.ru/inf/sorting.htm (in Russian) */
						l=0u;
						r=main_wdg.z_num-1u;
						while (l<r)
						{
							val=o_iter[med_z_xx];
							i=l;
							o_iter1=o_iter+i*main_wdg.x_num;
							j=r;
							o_iter2=o_iter+j*main_wdg.x_num;
							for ( ; ; )
							{
								for ( ; *o_iter1<val; o_iter1+=main_wdg.x_num,++i) ;
								for ( ; val<*o_iter2; o_iter2-=main_wdg.x_num,--j) ;
								if (i==j)
								{
									++i;
									--j;
									break;
								}
								if (i<j)
								{
									tmp=*o_iter1;
									*o_iter1=*o_iter2;
									*o_iter2=tmp;
									o_iter1+=main_wdg.x_num;
									o_iter2-=main_wdg.x_num;
									++i;
									--j;
								}
								if (i>j) break;
							}
							if (j<med_z) l=i;
							if (med_z<i) r=j;
						}
						*iter=o_iter[med_z_xx];
						++iter;
					}
				}
				free(data);
				break;
			}
			default/*2u*/: // user defined message length
				start0=usr_z_num;
				nc_get_vara_double(file1,matr1,start,count,matr);
				break;
		}
    }
    else
    {
		const char float_pt=txt_files->flt_pt;
		const char non_flt_pt=(float_pt=='.')? ',' : '.';
		FILE *file1=txt_files->v_file;
		const fpos_t matr1=txt_files->matr_v_pos;
		unsigned int i,j;
		const unsigned int x_num1=main_wdg.x_num-1u;
		Data_Text::Line l1;
		char *line1,*splitter,*pt_pos;
		double *iter=matr;

		switch (vals_for_edgs)
		{
			case 0u: // simple average
				memset(matr,0,xy_num*sizeof(double));
				fsetpos(file1,&matr1);
				for (unsigned int k=0u; k!=main_wdg.z_num; ++k)
				{
					iter=matr;
					for (j=0u; j!=main_wdg.x_num; ++j)
					{
						Data_Text::readline(file1,l1);
						line1=l1.Give_mdf();
						for (i=0u; i!=x_num1; ++i)
						{
							*(splitter=strchr(line1,'\t'))='\0';
							if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
								*pt_pos=float_pt;
							*iter+=atof(line1);
							++iter;
							line1=splitter+1;
						}
						if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
							*pt_pos=float_pt;
						*iter+=atof(line1);
						++iter;
					}
					while (Data_Text::readline(file1,l1))
					{
						if (strstr(l1.Give(),"Message length ")!=NULL) break;
					}
				}
				val=1.0/static_cast<double>(main_wdg.z_num);
				iter=matr;
				for (const double *iter_end=matr+xy_num; iter!=iter_end; ++iter)
					*iter*=val;
				break;
			case 1u: // median
			{
				const unsigned int med_z=(main_wdg.z_num-1u)>>1u;
				const unsigned int med_z_xx=med_z*main_wdg.x_num;
				double *data=static_cast<double*>(malloc(main_wdg.x_num*main_wdg.z_num*sizeof(double))); // aux
				unsigned int l,r;
				double *o_iter,*o_iter1,*o_iter2;
				const double *o_iter_end;
				double tmp;

				if (data==NULL) { NOT_ENOUGH_MEM_RET(false); } // too bad
				for (unsigned int k=0u; k!=main_wdg.x_num; ++k)
				{
					o_iter=data;
					fsetpos(file1,&matr1);
					for (j=0u; j!=main_wdg.z_num; ++j)
					{
						if (j!=0u)
						{
							while (Data_Text::readline(file1,l1))
							{
								if (strstr(l1.Give(),"Message length ")!=NULL) break;
							}
						}
						for (i=0u; i!=k; ++i) // skip (k-1) lines...
							Data_Text::readline(file1,l1);
						// ... and read line number 'k'
						Data_Text::readline(file1,l1);
						line1=l1.Give_mdf();
						for (i=0u; i!=x_num1; ++i)
						{
							*(splitter=strchr(line1,'\t'))='\0';
							if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
								*pt_pos=float_pt;
							*o_iter=atof(line1);
							++o_iter;
							line1=splitter+1;
						}
						if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
							*pt_pos=float_pt;
						*o_iter=atof(line1);
						++o_iter;
					}
					for (o_iter=data,o_iter_end=data+main_wdg.x_num; o_iter!=o_iter_end; ++o_iter)
					{
						/* the algorithm was taken from http://markx.narod.ru/inf/sorting.htm (in Russian) */
						l=0u;
						r=main_wdg.z_num-1u;
						while (l<r)
						{
							val=o_iter[med_z_xx];
							i=l;
							o_iter1=o_iter+i*main_wdg.x_num;
							j=r;
							o_iter2=o_iter+j*main_wdg.x_num;
							for ( ; ; )
							{
								for ( ; *o_iter1<val; o_iter1+=main_wdg.x_num,++i) ;
								for ( ; val<*o_iter2; o_iter2-=main_wdg.x_num,--j) ;
								if (i==j)
								{
									++i;
									--j;
									break;
								}
								if (i<j)
								{
									tmp=*o_iter1;
									*o_iter1=*o_iter2;
									*o_iter2=tmp;
									o_iter1+=main_wdg.x_num;
									o_iter2-=main_wdg.x_num;
									++i;
									--j;
								}
								if (i>j) break;
							}
							if (j<med_z) l=i;
							if (med_z<i) r=j;
						}
						*iter=o_iter[med_z_xx];
						++iter;
					}
				}
				free(data);
				break;
			}
			default/*2u*/: // user defined message length
				fsetpos(file1,&matr1);
				if (usr_z_num!=0u)
				{
					splitter=static_cast<char*>(malloc(50u*sizeof(char)));
					if (splitter==NULL) { NOT_ENOUGH_MEM_RET(false); } // nonsense!
					sprintf(splitter,"Message length %d",
							static_cast<int>(usr_z_num)*step_length+begin_message_length);
					for ( ; ; )
					{
						Data_Text::readline(file1,l1);
						if (strstr(l1.Give(),splitter)!=NULL) break; // at last we must be here!
					}
					free(splitter);
				}
				for (j=0u; j!=main_wdg.x_num; ++j)
				{
					Data_Text::readline(file1,l1);
					line1=l1.Give_mdf();
					for (i=0u; i!=x_num1; ++i)
					{
						*(splitter=strchr(line1,'\t'))='\0';
						if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
							*pt_pos=float_pt;
						*iter=atof(line1);
						++iter;
						line1=splitter+1;
					}
					if ((pt_pos=strchr(line1,non_flt_pt))!=NULL)
						*pt_pos=float_pt;
					*iter=atof(line1);
					++iter;
				}
				break;
		}
    }
    return true;
}
void TVWidget::SaveImageMenu (){
    QWidget *window = new QWidget;

    QHBoxLayout *layout = new QHBoxLayout;

    save_width->setParent(this);
    save_heigth->setParent(this);
    save_width->setMaximum(10000);
    save_heigth->setMaximum(10000);
    save_menu_btn->setParent(this);
    save_menu_btn->setText(tr("Save!"));
    layout->addWidget(save_width);
    layout->addWidget(save_heigth);
    layout->addWidget(save_menu_btn);
    window->setLayout(layout);
    connect(save_menu_btn,SIGNAL(clicked()),this,SLOT(SaveImage()));
    window->show();
}

bool TVWidget::MapGraphInto3D (double *matr, const double m_d_impact,
							   unsigned int &edg_n, unsigned int &edg50_n, unsigned int &edg99_n) {
	clock_t st=clock();

	unsigned int i,j,v,k,k1;
	double *mtr1=matr;
	double val;
	double min_val=DBL_MAX;

	for (i=0u,k=0u; i!=x_num; ++i,k+=x_num)
	{
		j=i+1u;
		k1=j*x_num+i;
		for (mtr1+=j; j!=x_num; ++j,++mtr1,k1+=x_num)
		{
			v=edge_counts[k+j]+edge_counts[k1];
			if (v==0u)
				*mtr1=0.0; // assign something
			else
			{
				// magic formula!
				val=matr[k1];
				val=val+(*mtr1-val)*static_cast<double>(edge_counts[k+j])/static_cast<double>(v);
				val=(val<1.0e-15)? 1.0e-15 : val;
				min_val=(val<min_val)? val : min_val;
				*mtr1=val;
			}
		}
	}
	min_val=1.0/(min_val*min_val);
	mtr1=matr;
	for (i=0u; i<x_num; ++i)
	{
		j=i+1u;
		for (mtr1+=j; j!=x_num; ++j,++mtr1)
		{
			// normalize and raise to the power of 2
			val=*mtr1;
			*mtr1=val*val*min_val;
		}
	}

	//! only the upper triangle of 'matr' is valid now!

	double *xx=static_cast<double*>(malloc(x_num*sizeof(double)));
	if (xx==NULL) return false;
	double *yy=static_cast<double*>(malloc(x_num*sizeof(double)));
	if (yy==NULL) { free(xx); return false; }
	double *zz=static_cast<double*>(malloc(x_num*sizeof(double)));
	if (zz==NULL) { free(yy); free(xx); return false; }

	/* do gradient descent */
	static const double eps=1.0e-100; // precision for distances and for the step
	static const double small_var=1.0e-6; // lower threshold of difference between
										  // previous and current value of minimizing function
	static const double dec_step=1.0/16.0,inc_step=1.5; // adjusting of the step
	static const unsigned int max_iter=300000u; // maximum number of iterations
	double *gradx=static_cast<double*>(malloc(x_num*sizeof(double)));
	if (gradx==NULL) { free(zz); free(yy); free(xx); return false; }
	double *grady=static_cast<double*>(malloc(x_num*sizeof(double)));
	if (grady==NULL) { free(gradx); free(zz); free(yy); free(xx); return false; }
	double *gradz=static_cast<double*>(malloc(x_num*sizeof(double)));
	if (gradz==NULL) { free(grady); free(gradx); free(zz); free(yy); free(xx); return false; }
	const unsigned int x_num1=x_num-1u;
	const double *mtr;
	const unsigned int *edg_cnt;
	double t=1.0e-7,f_prev,f;
	double x_i,y_i,z_i,dx,dy,dz,grx,gry,grz;
	const double half_m_d_impact=0.5*m_d_impact;

	srand(x_num);
	for (i=0u; i!=x_num; ++i)
	{
		xx[i]=static_cast<double>(i)*(1.0+static_cast<double>(rand())/static_cast<double>(RAND_MAX));
		yy[i]=static_cast<double>(i)*(1.0+static_cast<double>(rand())/static_cast<double>(RAND_MAX));
		zz[i]=static_cast<double>(i)*(1.0+static_cast<double>(rand())/static_cast<double>(RAND_MAX));
	}

	/* temporary add lower triangle to upper triangle;
	   this action makes access to 'edge_counts' in the descent cache friendly
	   (and speeds up each iteration by the ratio of 2 and greater when 'x_num'>2000)
	   do not forget to rollback these changes */
	for (i=0u,k=0u; i!=x_num1; ++i)
	{
		j=i+1u;
		k+=j;
		v=j*x_num+i;
		for ( ; j!=x_num; ++j,++k,v+=x_num)
			edge_counts[k]+=edge_counts[v];
	}
	/* move all necessary values in 'matr' closer
	   to the beginning to improve cache friendliness */
	mtr=mtr1=matr; // first row is OK
	edg_cnt=edge_counts;
	for (i=0u; i!=x_num1; ++i)
	{
		j=i+1u;
		mtr+=j;
		edg_cnt+=j;
		for ( ; j!=x_num; ++j,++mtr,++edg_cnt)
		{
			if (*edg_cnt!=0u)
			{
				*mtr1=*mtr;
				++mtr1;
			}
		}
	}

	f=0.0;
	mtr=matr;
	edg_cnt=edge_counts;
	for (i=0u,k=0u; i!=x_num1; ++i)
	{
		x_i=xx[i]; y_i=yy[i]; z_i=zz[i];
		j=i+1u;
		for (edg_cnt+=j; j!=x_num; ++j,++edg_cnt)
		{
			dx=x_i-xx[j]; dy=y_i-yy[j]; dz=z_i-zz[j];
			val=dx*dx+dy*dy+dz*dz;
			if (*edg_cnt==0u)
				f+=((val<eps)? m_d_impact : ((val<1.0)? (m_d_impact/val) : 0.0));
			else
			{
				val-=*mtr;
				f+=(val*val);
				++mtr;
			}
		}
	}
	f*=0.5;
	f_prev=f;

	QLabel l(this);
	l.setFixedSize(200,50);
	l.move((width()-l.width())/2,(height()-l.height())/2);
	l.setAutoFillBackground(true);
	l.show();

	clock_t st1;
	for (k=0u; k!=max_iter; ++k)
	{
		if ((k & 0x1f)==0u)
		{
			l.setText(QString("<div align=\"center\">iter %1 / %2</div><br><div align=\"left\">func: %3</div>").
					  arg(k).arg(max_iter).arg(f,0,'g',12));
			// immediate processing of all paint events and such
			QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents,1);
		}
		st1=clock();
		memset(gradx,0,x_num*sizeof(double));
		memset(grady,0,x_num*sizeof(double));
		memset(gradz,0,x_num*sizeof(double));
		f=0.0;
		t=-t; // temporary
		mtr=matr;
		edg_cnt=edge_counts;
		for (i=0u; i<x_num1; ++i)
		{
			x_i=xx[i]; y_i=yy[i]; z_i=zz[i];
			grx=gry=grz=0.0;
			j=i+1u;
			for (edg_cnt+=j; j<x_num; ++j,++edg_cnt)
			{
				dx=x_i-xx[j]; dy=y_i-yy[j]; dz=z_i-zz[j];
				val=dx*dx+dy*dy+dz*dz;
				if (*edg_cnt==0u)
				{
					if (!(val<1.0)) continue;
					if (val<eps)
					{
						dx=dy=dz=-half_m_d_impact;
					}
					else
					{
						val=-m_d_impact/(val*val);
						dx*=val; dy*=val; dz*=val;
					}
				}
				else
				{
					(val-=*mtr)*=2.0;
					++mtr;
					dx*=val; dy*=val; dz*=val;
				}
				grx+=dx; gry+=dy; grz+=dz;
				gradx[j]-=dx; grady[j]-=dy; gradz[j]-=dz;
			}
			gradx[i]+=grx; grady[i]+=gry; gradz[i]+=grz;
		}
		for (i=0u; i<x_num; ++i)
		{
			gradx[i]*=t;
			xx[i]+=gradx[i];
			grady[i]*=t;
			yy[i]+=grady[i];
			gradz[i]*=t;
			zz[i]+=gradz[i];
		}
		mtr=matr;
		edg_cnt=edge_counts;
		for (i=0u; i<x_num1; ++i)
		{
			x_i=xx[i]; y_i=yy[i]; z_i=zz[i];
			j=i+1u;
			for (edg_cnt+=j; j<x_num; ++j,++edg_cnt)
			{
				dx=x_i-xx[j]; dy=y_i-yy[j]; dz=z_i-zz[j];
				val=dx*dx+dy*dy+dz*dz;
				if (*edg_cnt==0u)
					f+=((val<eps)? m_d_impact : ((val<1.0)? (m_d_impact/val) : 0.0));
				else
				{
					val-=*mtr;
					f+=(val*val);
					++mtr;
				}
			}
		}
		f*=0.5;
		t=-t;
		st1=clock()-st1;
		printf("  %u: prev=%.12g cur=%.12g t=%g, %g мс\n",k,f_prev,f,t,
				static_cast<double>(st1*1000u)/static_cast<double>(CLOCKS_PER_SEC));
		if (f>f_prev)
		{
			if (f<f_prev+small_var) break;
			for (i=0u; i!=x_num; ++i) // return old values
		   	{
		   		xx[i]-=gradx[i];
				yy[i]-=grady[i];
				zz[i]-=gradz[i];
			}
			t*=dec_step; // decrease the step
			if (t<eps) break; // the step is too small
		}
		else
		{
			if (f_prev<f+small_var) break;
			f_prev=f;
			t*=inc_step; // increase the step
		}
	}
	f=(f>f_prev)? f_prev : f;
	free(gradz);
	free(grady);
	free(gradx);

	l.hide();

	st=clock()-st;

	printf("\nReached optimum: %.12g\n\n",f);

	printf("\nEND BUILD GRAPH: %g мс\n\n",static_cast<double>(st*1000u)/static_cast<double>(CLOCKS_PER_SEC));

	/* count errors */
	unsigned int edg_num=0u; // number of all edges
	unsigned int edg50_num=0u; // number of edges with length error not less than 50%
	unsigned int edg99_num=0u; // number of edges with length error not less than 99%

	mtr=matr;
	for (i=0u,v=0u; i!=x_num1; ++i)
	{
		x_i=xx[i]; y_i=yy[i]; z_i=zz[i];
		j=i+1u;
		for (v+=j; j!=x_num; ++j,++v)
		{
			if (edge_counts[v]==0u)
				continue;
			++edg_num;
			dx=x_i-xx[j]; dy=y_i-yy[j]; dz=z_i-zz[j];
			val=fabs(dx*dx+dy*dy+dz*dz-*mtr);
			if (val>=*mtr*0.25) // squares are compared, that's why 0.25 instead of 0.5
			{
				++edg50_num;
				if (val>=*mtr*0.9801)
					++edg99_num;
			}
			++mtr;
		}
	}
	edg_n=edg_num;
	edg50_n=edg50_num;
	edg99_n=edg99_num;

	// rollback the changes
	for (i=0u,k=0u; i!=x_num1; ++i)
	{
		j=i+1u;
		k+=j;
		v=j*x_num+i;
		for ( ; j!=x_num; ++j,++k,v+=x_num)
			edge_counts[k]-=edge_counts[v];
	}

	/* centre the graph */
	double min_x=DBL_MAX,max_x=1.0-DBL_MAX,min_y=DBL_MAX,max_y=1.0-DBL_MAX,min_z=DBL_MAX,max_z=1.0-DBL_MAX;
	for (i=0u; i!=x_num; ++i)
	{
		min_x=(xx[i]<min_x)? xx[i] : min_x;
		max_x=(xx[i]>max_x)? xx[i] : max_x;
		min_y=(yy[i]<min_y)? yy[i] : min_y;
		max_y=(yy[i]>max_y)? yy[i] : max_y;
		min_z=(zz[i]<min_z)? zz[i] : min_z;
		max_z=(zz[i]>max_z)? zz[i] : max_z;
	}
	min_x=-(min_x+max_x)*0.5;
	min_y=-(min_y+max_y)*0.5;
	geom_c_z=(min_z+max_z)*0.5;
	min_z=5.0-min_z; // the nearest (to viewer) z-coordinate should be equal to 5
	geom_c_z+=min_z;
	for (i=0u; i!=x_num; ++i)
	{
		xx[i]+=min_x;
		yy[i]+=min_y;
		zz[i]+=min_z;
	}

	/* convert coordinates from double to float */
	double *arr_end;
	float *arr;

	/* there is enough memory for these 3 arrays
	   because 'gradx', 'grady' and 'gradz' were free'd */
	arr=static_cast<float*>(malloc(x_num*sizeof(float)));
	points_z=arr;
	for (arr_end=zz+x_num; zz!=arr_end; ++zz,++arr)
		*arr=static_cast<float>(*zz);
	zz-=x_num;
	free(zz);
	arr=static_cast<float*>(malloc(x_num*sizeof(float)));
	points_y=arr;
	for (arr_end=yy+x_num; yy!=arr_end; ++yy,++arr)
		*arr=static_cast<float>(*yy);
	yy-=x_num;
	free(yy);
	arr=static_cast<float*>(malloc(x_num*sizeof(float)));
	points_x=arr;
	for (arr_end=xx+x_num; xx!=arr_end; ++xx,++arr)
		*arr=static_cast<float>(*xx);
	xx-=x_num;
	free(xx);

	return true;
}

void TopologyViewer::CompareTopologies (void) {
	if (hosts_undefined)
	{
		SEND_ERR_MSG(NoTopoHosts);
		return;
	}

	QString ideal_topo_fname(QFileDialog::getOpenFileName(this,tr("Open file with \"ideal\" topology")));

	if (ideal_topo_fname.isEmpty()) return;

	// without this line file dialog remains wisible until the end (!) of the function
	QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents,20);

	FILE *ideal_topo_file=fopen(ideal_topo_fname.toLocal8Bit().constData(),"r");

	if (ideal_topo_file==NULL)
	{
		SEND_ERR_MSG1(CannotOpen,ideal_topo_fname);
		return;
	}

	Data_Text::Line l;
	const char *line1;

	for ( ; ; )
	{
		if (!Data_Text::readline(ideal_topo_file,l))
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(UnexpEOF,ideal_topo_fname);
			return;
		}
		for (line1=l.Give(); isspace(*line1); ++line1) ;
		if (*line1!='\0') break;
	}
	if (strstr(line1,"graph")!=line1)
	{
		fclose(ideal_topo_file);
		SEND_ERR_MSG1(NoLexGraph,ideal_topo_fname);
		return;
	}
	for (line1=l.Give()+(strlen(l.Give())-1u); isspace(*line1); --line1) ; // 'line1' cannot be equal to 'l.Give()'
	if (*line1!='{')
	{
		// '{' symbol is not the last visible symbol in 'l.Give()'
		fclose(ideal_topo_file);
		SEND_ERR_MSG1(BraceNotLast,ideal_topo_fname);
		return;
	}

	// this map is put into operation to reorder 'host_names' for faster search;
	// hosts' names are not unique what means that one name corresponds to several processes
	// (for example, cores of a single processor have the same names)
	std::map<QString,std::pair<std::vector<unsigned int>,unsigned int> > our_hosts;

	unsigned int i;
	std::map<QString,std::pair<std::vector<unsigned int>,unsigned int> >::iterator o_h_it,o_h_it_end;

	try {
		for (i=0u; i!=main_wdg.x_num; ++i)
			our_hosts[main_wdg.host_names[i]].first.push_back(i);
	}
	catch (...)
	{
		fclose(ideal_topo_file);
		NOT_ENOUGH_MEM_RET();
	}
	o_h_it_end=our_hosts.end(); // and it won't be changed
	for (o_h_it=our_hosts.begin(); o_h_it!=o_h_it_end; ++o_h_it)
		o_h_it->second.second=0u; // this is a number of accesses to '*o_h_it' during searches

	// "ideal" topology graph (directed!);
	// each vertex has a name and an array of adjacent vertices;
	// the map is used for faster search of duplicate edges;
	// the second elements of the map store weights of edges
	// (reciprocal to bandwidth of that channel, that is number of nanoseconds to send 1 bit)
	std::vector<std::pair<QString,std::map<unsigned int,double> > > ideal_topo;

	static const unsigned int UINT_MAX_DIV_10=UINT_MAX / 10u;
	static const char UINT_MAX_MOD_10=static_cast<const char>(UINT_MAX % 10u);

	unsigned int v1,v2;
	QString name;

	for ( ; ; )
	{
		/* collect vertices of "ideal" topology graph;
		   each vertex must be written in the folowing format:
		   'v123 [label="name"];',
		   where 'v', '[]', 'label=' and ';' are necessary elements */
		if (!Data_Text::readline(ideal_topo_file,l))
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(UnexpEOF,ideal_topo_fname);
			return;
		}
		line1=l.Give();
		/* find vertex ID */
		for ( ; isspace(*line1); ++line1) ;
		if (*line1!='v')
		{
			if (*line1=='\0')
				continue;
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(InvLexVertex,ideal_topo_fname);
			return;
		}
		++line1;
		if (!isdigit(*line1))
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(InvLexVertex,ideal_topo_fname);
			return;
		}
		/* collect vertex index */
		v1=static_cast<unsigned int>(*line1-'0');
		for (++line1; isdigit(*line1); ++line1)
		{
			if ((v1>UINT_MAX_DIV_10) || ((v1==UINT_MAX_DIV_10) && ((*line1-'0')>UINT_MAX_MOD_10)))
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(OverflIndVert,ideal_topo_fname);
				return;
			}
			(v1*=10u)+=static_cast<unsigned int>(*line1-'0');
		}
		/* find the construct '[label="' */
		for ( ; isspace(*line1); ++line1) ;
		if (*line1!='[')
		{
			if ((*line1=='-') && (*(line1+1)=='-'))
				break; // we found an edge!
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoLexOpenBracket,ideal_topo_fname);
			return;
		}
		for (++line1; isspace(*line1); ++line1) ;
		if (strstr(line1,"label")!=line1)
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoLexLabel,ideal_topo_fname);
			return;
		}
		for (line1+=5; isspace(*line1); ++line1) ;
		if (*line1!='=')
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoLexLabel,ideal_topo_fname);
			return;
		}
		for (++line1; isspace(*line1); ++line1) ;
		if (*line1!='\"')
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoLexFrwQuote,ideal_topo_fname);
			return;
		}
		/* collect vertex name */
		name.clear();
		try {
			for (++line1; (*line1!='\0') && (*line1!='\"'); ++line1)
				name+=*line1;
		}
		catch (...)
		{
			fclose(ideal_topo_file);
			NOT_ENOUGH_MEM_RET();
		}
		/* find the construct '"];' */
		if (*line1!='\"')
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoLexBckwQuote,ideal_topo_fname);
			return;
		}
		for (++line1; (*line1!='\0') && (*line1!=']'); ++line1) ;
		if (*line1!=']')
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoLexCloseBracket,ideal_topo_fname);
			return;
		}
		for (++line1; (*line1!='\0') && (*line1!=';'); ++line1) ;
		if (*line1!=';')
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoLexSemicolon,ideal_topo_fname);
			return;
		}
		for (++line1; (*line1!='\0') && isspace(*line1); ++line1) ;
		if (*line1!='\0')
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(SemicolonNotLast,ideal_topo_fname);
			return;
		}
		o_h_it=our_hosts.find(name);
		if (o_h_it!=o_h_it_end)
			++(o_h_it->second.second); // plus 1 hit to that name
		/* add new vertex */
		if (ideal_topo.size()<=v1)
		{
			try {
				ideal_topo.resize(v1+1u);
			}
			catch (...) {
				fclose(ideal_topo_file);
				NOT_ENOUGH_MEM_RET();
			}
		}
		ideal_topo[v1].first=name;
	}
	/* check if all elements in 'our_hosts' was hit strictly once */
	for (o_h_it=our_hosts.begin(); o_h_it!=o_h_it_end; ++o_h_it)
	{
		if (o_h_it->second.second==0u)
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(NoMatchingTopo,ideal_topo_fname);
			return;
		}
		if (o_h_it->second.second!=1u)
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(DuplicateNames,ideal_topo_fname);
			return;
		}
	}

	std::pair<unsigned int,double> sec_end; // the second end and a reciprocal to a bandwidth of an edge
	std::pair<std::map<unsigned int,double>::iterator,bool> i_tp_ins; // result of insertion into 'ideal_topo'
	char *line;

	for ( ; ; )
	{
		/* collect edges of "ideal" topology graph;
		   each edge must be written in the folowing format:
		   'v123 -- v456 [label="789Gbit/s"];',
		   where 'v', '--', '[]', 'label=', 'Gbit/s' and ';' are necessary elements */
		line1=l.Give();
		for ( ; isspace(*line1); ++line1) ;
		if (*line1!='\0')
		{
			if (*line1=='}')
				// the "ideal" topology was read successfully!
				break;
			/* find first vertex ID */
			if (*line1!='v')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(InvLexVertex,ideal_topo_fname);
				return;
			}
			++line1;
			if (!isdigit(*line1))
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(InvLexVertex,ideal_topo_fname);
				return;
			}
			/* collect first vertex index */
			v1=static_cast<unsigned int>(*line1-'0');
			for (++line1; isdigit(*line1); ++line1)
			{
				if ((v1>UINT_MAX_DIV_10) || ((v1==UINT_MAX_DIV_10) && ((*line1-'0')>UINT_MAX_MOD_10)))
				{
					fclose(ideal_topo_file);
					SEND_ERR_MSG1(OverflIndEdge,ideal_topo_fname);
					return;
				}
				(v1*=10u)+=static_cast<unsigned int>(*line1-'0');
			}
			if (v1>=ideal_topo.size())
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(InvVertInEdge,ideal_topo_fname);
				return;
			}
			/* find '--' */
			for ( ; isspace(*line1); ++line1) ;
			if ((*line1!='-') || (*(line1+1)!='-'))
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoEdgeMark,ideal_topo_fname);
				return;
			}
			/* find second vertex ID */
			for (line1+=2; isspace(*line1); ++line1) ;
			if (*line1!='v')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(InvLexVertex,ideal_topo_fname);
				return;
			}
			++line1;
			if (!isdigit(*line1))
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(InvLexVertex,ideal_topo_fname);
				return;
			}
			/* collect second vertex index */
			v2=static_cast<unsigned int>(*line1-'0');
			for (++line1; isdigit(*line1); ++line1)
			{
				if ((v2>UINT_MAX_DIV_10) || ((v2==UINT_MAX_DIV_10) && ((*line1-'0')>UINT_MAX_MOD_10)))
				{
					fclose(ideal_topo_file);
					SEND_ERR_MSG1(OverflIndEdge,ideal_topo_fname);
					return;
				}
				(v2*=10u)+=static_cast<unsigned int>(*line1-'0');
			}
			if (v2>=ideal_topo.size())
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(InvVertInEdge,ideal_topo_fname);
				return;
			}
			if (v1==v2)
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(SelfLoopEdge,ideal_topo_fname);
				return;
			}
			/* find the construct '[label="' */
			for ( ; isspace(*line1); ++line1) ;
			if (*line1!='[')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexOpenBracket,ideal_topo_fname);
				return;
			}
			for (++line1; isspace(*line1); ++line1) ;
			if (strstr(line1,"label")!=line1)
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexLabel,ideal_topo_fname);
				return;
			}
			for (line1+=5; isspace(*line1); ++line1) ;
			if (*line1!='=')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexLabel,ideal_topo_fname);
				return;
			}
			for (++line1; isspace(*line1); ++line1) ;
			if (*line1!='\"')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexFrwQuote,ideal_topo_fname);
				return;
			}
			++line1;
			sec_end.second=strtod(line1,&line);
			line1=line;
			if (sec_end.second<3.0e-308)
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(ZeroBandw,ideal_topo_fname);
				return;
			}
			sec_end.second=1.0/sec_end.second; // turn bandwidth into time
			for (++line1; isspace(*line1); ++line1) ;
			if (strstr(line1,"Gbit/s")!=line1)
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexBits,ideal_topo_fname);
				return;
			}
			/* find the construct '"];' */
			for (line1+=6; isspace(*line1); ++line1) ;
			if (*line1!='\"')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexBckwQuote,ideal_topo_fname);
				return;
			}
			for (++line1; (*line1!='\0') && (*line1!=']'); ++line1) ;
			if (*line1!=']')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexCloseBracket,ideal_topo_fname);
				return;
			}
			for (++line1; (*line1!='\0') && (*line1!=';'); ++line1) ;
			if (*line1!=';')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(NoLexSemicolon,ideal_topo_fname);
				return;
			}
			for (++line1; (*line1!='\0') && isspace(*line1); ++line1) ;
			if (*line1!='\0')
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(SemicolonNotLast,ideal_topo_fname);
				return;
			}
			sec_end.first=v2;
			try {
				i_tp_ins=ideal_topo[v1].second.insert(sec_end);
			}
			catch (...)
			{
				fclose(ideal_topo_file);
				NOT_ENOUGH_MEM_RET();
			}
			if (!i_tp_ins.second)
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(DuplicateEdge,ideal_topo_fname);
				return;
			}
			sec_end.first=v1;
			try {
				i_tp_ins=ideal_topo[v2].second.insert(sec_end);
			}
			catch (...)
			{
				fclose(ideal_topo_file);
				NOT_ENOUGH_MEM_RET();
			}
			if (!i_tp_ins.second)
			{
				fclose(ideal_topo_file);
				SEND_ERR_MSG1(DuplicateEdge,ideal_topo_fname);
				return;
			}
		}
		if (!Data_Text::readline(ideal_topo_file,l))
		{
			fclose(ideal_topo_file);
			SEND_ERR_MSG1(UnexpEOF,ideal_topo_fname);
			return;
		}
	}
	l.Destroy();
	fclose(ideal_topo_file);

	double similarity=0.0; // percent of per-edge similarity between "real" and "ideal" topology
	unsigned int edges_num_in_ideal=0u; // number of edges in "ideal" topology which connect
										// vertices in "real" topology
	std::map<unsigned int,double>::const_iterator it,it_end;
	std::vector<unsigned int>::const_iterator it2,it2_end,it3,it3_end;
	unsigned int max_edg_cnt;
	const unsigned int n=ideal_topo.size();
	unsigned int *edg_cnt;

	for (i=0u; i!=n; ++i)
	{
		o_h_it=our_hosts.find(ideal_topo[i].first);
		if (o_h_it==o_h_it_end) continue;
		const std::vector<unsigned int> &from=o_h_it->second.first;
		for (it=ideal_topo[i].second.begin(),it_end=ideal_topo[i].second.end(); it!=it_end; ++it)
		{
			o_h_it=our_hosts.find(ideal_topo[it->first].first);
			if (o_h_it==o_h_it_end) continue;
			++edges_num_in_ideal;
			// processor with ID 'it->first' in "ideal" topology corresponds
			// to the whole group of processor cores in "real" topology
			const std::vector<unsigned int> &to=o_h_it->second.first;
			max_edg_cnt=0u;
			for (it2=from.begin(),it2_end=from.end(); it2!=it2_end; ++it2)
			{
				edg_cnt=main_wdg.edge_counts+(*it2*main_wdg.x_num);
				for (it3=to.begin(),it3_end=to.end(); it3!=it3_end; ++it3)
					// choose maximum probability
					max_edg_cnt=(max_edg_cnt<edg_cnt[*it3])? edg_cnt[*it3] : max_edg_cnt;
			}
			similarity+=static_cast<double>(max_edg_cnt); // first part of magic formula!
		}
	}
	// note that 'edges_num_in_ideal' is two times greater than the real number of edges
	// because "ideal" topology is undirected graph and we store it like directed graph
	if (edges_num_in_ideal!=0u)
	{
		// "ideal" topology is regular (grids, hypercubes, toruses etc...)
		similarity*=(100.0/static_cast<double>(main_wdg.z_num*edges_num_in_ideal)); // second part of magic formula!
		if (similarity!=0.0) // precise 0
		{
			QMessageBox::information(this,tr("Comparison of topologies"),
									 tr("The topology from file<br><i>%1</i><br>is similar to the "
									 	"retrieved topology by <b>%2%</b>").arg(ideal_topo_fname).arg(similarity));
			return;
		}
	}

	// "ideal" topology is hierarchical (trees, stars etc...)

	/* show a red-blue representation of "ideal" (!) topology */

	// this structure represents a list of weighted edges;
	// each edge is a map 'v1'->'v2', where 'v1' and 'v2' are vertices;
	// (note that 'v1' is strongly less than 'v2');
	// a weight of an edge is a simple average of some values;
	// to compute a simple average a {double,int} pair is used;
	// useful information: 'real_times' is a subset of 'ideal_topo'
	std::map<unsigned int,std::map<unsigned int,std::pair<double,unsigned int> > > real_times;

	double *matr; // values for "real" topology
	double ml; // message length (or a simple average of all message lengths depending on 'vals_for_edgs')

	matr=static_cast<double*>(malloc(main_wdg.x_num*main_wdg.x_num*sizeof(double)));
	if (matr==NULL) { NOT_ENOUGH_MEM_RET(); }
	if (!GetMatrixByValsForEdgsVar(matr))
	{
		free(matr);
		NOT_ENOUGH_MEM_RET();
	}
	switch (vals_for_edgs)
	{
		case 0u: // simple average
			ml=0.5*static_cast<double>((this->begin_message_length<<1u)+
										static_cast<int>(main_wdg.z_num-1u)*this->step_length);
			break;
		case 1u: // median
			ml=static_cast<double>(this->begin_message_length+
								   static_cast<int>(main_wdg.z_num>>1u)*this->step_length);
			break;
		default /*2u*/: // user defined
			ml=static_cast<double>(static_cast<int>(usr_z_num)*this->step_length+this->begin_message_length);
			break;
	}
	ml=(ml<DBL_EPSILON)? 1.0 : ml; // move away from zero to keep edges' weights positive

	unsigned int *previous; // reverse of shortest path between 2 vertices in "ideal" topology
	bool *visited; // array of indicators for vertices
	double *dist; // array of distances to vertices
	unsigned int i1,k;
	double min_dist;
	double time_real; // weight of an edge in "real" topology
	double r_sum_tm; // 1.0 divided by a length of a path between two vertices
	std::map<unsigned int,std::pair<double,unsigned int> > *e2; // right part of the first map in 'real_times'
	std::pair<double,unsigned int> *e1e2; // right part of the map in 'e2'
	unsigned int vb,ve;

	dist=static_cast<double*>(malloc(n*sizeof(double)));
	if (dist==NULL) { free(matr); NOT_ENOUGH_MEM_RET(); }
	previous=static_cast<unsigned int*>(malloc(n*sizeof(int)));
	if (previous==NULL) { free(dist); free(matr); NOT_ENOUGH_MEM_RET(); }
	visited=static_cast<bool*>(malloc(n*sizeof(bool)));
	if (visited==NULL) { free(previous); free(dist); free(matr); NOT_ENOUGH_MEM_RET(); }
	printf("\n");for (i=0u; i!=n; ++i)
	{
		o_h_it=our_hosts.find(ideal_topo[i].first);
		if (o_h_it==o_h_it_end) continue;
		const std::vector<unsigned int> &from=o_h_it->second.first;
		for (i1=0u; i1!=n; ++i1)
		{
			// find a pair of processes contained by both "real" and "ideal" topologies
			if (i1==i) continue;
			o_h_it=our_hosts.find(ideal_topo[i1].first);
			if (o_h_it==o_h_it_end) continue;
			// find the shortest path from 'i' to 'i1';
			// Dijkstra's algorithm is used
			for (v1=0u; v1!=n; ++v1)
			{
				previous[v1]=UINT_MAX;
				dist[v1]=DBL_MAX;
			}
			memset(visited,0,n*sizeof(bool));
			dist[i]=0.0;
			for (v2=0u; v2!=n; ++v2)
			{
				for (v1=0u; visited[v1]; ++v1) ;
				min_dist=dist[v1];
				for (k=v1+1u; k!=n; ++k)
				{
					if (!visited[k] && (min_dist>dist[k]))
					{
						v1=k;
						min_dist=dist[k];
					}
				}
				if (min_dist==DBL_MAX) break; // fail
				if (v1==i1) break; // the shortest path is found
				visited[v1]=true;
				for (it=ideal_topo[v1].second.begin(),it_end=ideal_topo[v1].second.end(); it!=it_end; ++it)
				{
					if (visited[it->first]) continue;
					min_dist=dist[v1]+it->second;
					if (dist[it->first]>min_dist)
					{
						dist[it->first]=min_dist;
						previous[it->first]=v1;
					}
				}
			}
			v2=i1;
			v1=previous[v2];
			if (v1==UINT_MAX) break; // 'i' and 'i1' are not connected

			// so, we have a weighted path from 'i' to 'i1' (direction is important!)
			// in "ideal" topology and a measure of time for this pair of vertices in "real" topology;
			//
			// let 'p' be the path and 't' be the measure of time; let 'p' consists of 'pn' edges
			// 'p[0]', 'p[1]', ..., 'p['pn'-1]'; and let 'tp[j]' be the length of the edge 'p[j]';
			// we can distribute 't' to edge 'p[j]' using the following formula: t*tp[j] / (tp[0]+...+tp[pn-1]),
			// i.e. the more 'tp[j]' is, the bigger part of 't' is given to 'p[j]';
			// let 'e1' and 'e2' be the two ends of an edge 'p[j]', then add the result of the
			// mentioned formula to real_times[e1][e2].first and add 1 to real_times[e1][e2].second;
			// in general, more than one path containes 'p[j]', so a simple average of collected
			// values is used;
			//
			// let's designate 'tp_i' as (real_times[e1][e2].first/real_times[e1][e2].second);
			//
			// a group of edges in "real" topology corresponds to a single edge in "ideal" topology,
			// so 't' is a simple average of weights of edges in that group (the weights depend on
			// 'vals_for_edgs' variable) and 'ml' as message length (or a simple average of
			// all message lengths depending on 'vals_for_edgs');

			// compute 1/('tp[0]'+...+'tp[pn-1]')
			printf("\r111111: %u->%u",i,i1);fflush(stdout);r_sum_tm=0.0;
			for ( ; ; )
			{
				r_sum_tm+=ideal_topo[v1].second[v2];
				if (v1==i) break;
				v2=v1;
				v1=previous[v2];
			}
			r_sum_tm=1.0/r_sum_tm;
			// compute 't'/('tp[0]'+...+'tp[pn-1]')
			time_real=0.0;
			const std::vector<unsigned int> &to=o_h_it->second.first;
			for (it2=from.begin(),it2_end=from.end(); it2!=it2_end; ++it2)
			{
				v1=*it2*main_wdg.x_num;
				for (it3=to.begin(),it3_end=to.end(); it3!=it3_end; ++it3)
					time_real+=matr[v1+*it3];
			}
			(time_real*=r_sum_tm)/=static_cast<double>(from.size()*to.size());
			// distribute the result
			v2=i1;
			v1=previous[v2];
			for ( ; ; )
			{
				vb=(v1<v2)? v1 : v2; // begin vertex
				ve=(v1<v2)? v2 : v1; // end vertex
				try {
					e2=&(real_times[vb]); // new element can be created
				}
				catch (...) { free(visited); free(previous); free(dist); free(matr); NOT_ENOUGH_MEM_RET(); }
				k=e2->size();
				try {
					e1e2=&((*e2)[ve]); // new element can be created
				}
				catch (...) { free(visited); free(previous); free(dist); free(matr); NOT_ENOUGH_MEM_RET(); }
				if (k==e2->size())
				{
					// edge 'vb'->'ve' has already been presented in 'real_times'
					e1e2->first+=(time_real*ideal_topo[vb].second[ve]);
					++(e1e2->second);
				}
				else
				{
					// new edge
					e1e2->first=time_real*ideal_topo[vb].second[ve];
					e1e2->second=1u;
				}
				if (v1==i) break;
				v2=v1;
				v1=previous[v2];
			}
		}
	}
	printf("\n");free(visited);
	free(previous);
	free(dist);
	free(matr);

	if (real_times.empty())
	{
		// it means that "ideal" topology does not provide useful information:
		// all vertices we are interested in are unconnected
		emit SendMessToLog(MainWindow::Info,my_sign,ErrMsgs::ToString(NV::UselessTopoMatch,1,&ideal_topo_fname));
		return;
	}

	std::map<unsigned int,std::map<unsigned int,std::pair<double,unsigned int> > >::iterator r_t_it,r_t_it_end;
	std::map<unsigned int,std::pair<double,unsigned int> >::iterator it4,it4_end;
	std::map<unsigned int,double>::iterator it5,it5_end;

	/* finish calculations in 'real_times' */
	for (r_t_it=real_times.begin(),r_t_it_end=real_times.end(); r_t_it!=r_t_it_end; ++r_t_it)
	{
		for (it4=r_t_it->second.begin(),it4_end=r_t_it->second.end(); it4!=it4_end; ++it4)
			it4->second.first/=static_cast<double>(it4->second.second);
	}

	/* multiply all weights of edges in 'ideal_topo' by 'ml' */
	for (i=0u; i!=n; ++i)
	{
		for (it5=ideal_topo[i].second.begin(),it5_end=ideal_topo[i].second.end(); it5!=it5_end; ++it5)
			it5->second*=ml; // now 'it5->second' represents an "ideal" message pass
	}

	/* find minimum and maximum values in the intersection
	   of 'real_times' and 'ideal_topo' to paint edges */
	double gt_min,gt_max; // minimum and maximum of positive differencies for color mapping
	double lt_min,lt_max; // minimum and maximum of negative differencies for color mapping

	gt_min=DBL_MAX;
	gt_max=0.0;
	lt_min=DBL_MAX;
	lt_max=0.0;
	r_t_it=real_times.begin();
	for (i=0u; i!=n; ++i)
	{
		if (r_t_it==r_t_it_end) break;
		if (i!=r_t_it->first) continue;
		it=ideal_topo[i].second.lower_bound(i+1u);
		it_end=ideal_topo[i].second.end();
		it4=r_t_it->second.begin();
		it4_end=r_t_it->second.end();
		for ( ; ; )
		{
			if (it4==it4_end) break;
			if (it->first==it4->first)
			{
				// edge 'i'->'it->first' is coloured
				time_real=it->second-it4->second.first; // temporary
				if (time_real>0.0)
				{
					// for red color
					gt_min=(time_real<gt_min)? time_real : gt_min;
					gt_max=(gt_max<time_real)? time_real : gt_max;
				}
				else
				{
					// for blue color
					time_real=-time_real; // 'fchs' instruction in FPU
					lt_min=(time_real<lt_min)? time_real : lt_min;
					lt_max=(lt_max<time_real)? time_real : lt_max;
				}
				++it4;
			}
			++it;
		}
		++r_t_it;
	}
	gt_max-=gt_min; // turn 'gt_max' into color mapping coefficient
	gt_max=(gt_max<DBL_EPSILON)? 255.0 : 255.0/gt_max;
	lt_max-=lt_min; // turn 'lt_max' into color mapping coefficient
	lt_max=(lt_max<DBL_EPSILON)? 255.0 : 255.0/lt_max;

	// free some memory
	for (o_h_it=our_hosts.begin(); o_h_it!=o_h_it_end; ++o_h_it)
		o_h_it->second.first.clear();

	TVWidget *ideal_topo_view; // widget with red-blue representation of "ideal" topology graph

	try {
		ideal_topo_view=new TVWidget;
	}
	catch (...) { NOT_ENOUGH_MEM_RET(); }
	ideal_topo_view->x_num=n;
	ideal_topo_view->z_num=1u;

	quint8 *bit_arr; // iterator for bit arrays
	quint8 one_byte;
	unsigned int n1;

	/* determine colors of vertices in "ideal" topology */
	bit_arr=static_cast<quint8*>(malloc((n>>3u)+1u));
	if (bit_arr==NULL) { delete ideal_topo_view; NOT_ENOUGH_MEM_RET(); }
	ideal_topo_view->i_v_color=bit_arr;
	one_byte=0u;
	n1=0u;
	for (i=0u; i!=n; ++i)
	{
		if (our_hosts.find(ideal_topo[i].first)!=o_h_it_end)
			one_byte|=static_cast<quint8>(0x1<<n1);
		++n1;
		if (n1==8u)
		{
			n1=0u;
			*bit_arr=one_byte;
			++bit_arr;
			one_byte=0u;
		}
	}
	if ((n & 0x7)!=0u) *bit_arr=one_byte; // the last byte

	our_hosts.clear(); // no longer necessary

	/* compute the number of edges in "ideal" topology (remember that it is undirected) */
	n1=0u;
	for (i=0u; i!=n; ++i)
		n1+=ideal_topo[i].second.size();
	bit_arr=static_cast<quint8*>(calloc((n1>>3u)+1u,1u)); // n1 / 2
	if (bit_arr==NULL) { delete ideal_topo_view; NOT_ENOUGH_MEM_RET(); }
	ideal_topo_view->i_e_color=bit_arr;

	/* paint existing edges */
	unsigned char *clr_arr;

	n1=0u;
	for (r_t_it=real_times.begin(); r_t_it!=r_t_it_end; ++r_t_it)
		n1+=r_t_it->second.size();
	clr_arr=static_cast<unsigned char*>(malloc(n1*sizeof(char)));
	if (clr_arr==NULL) { delete ideal_topo_view; NOT_ENOUGH_MEM_RET(); }
	ideal_topo_view->i_e_color_val=clr_arr;

	// (all designations were taken from that big comment high above)
	//
	// 'tp[i]'*'ml' represents an "ideal" message pass and 'tp_i' represents a "real" message pass;
	// colors of edges are determined by comparing 'tp[i]'*'ml' and 'tp_i': if the former is greater
	// than the latter then the edge is painted in red else the edge is painted in blue (very rare case!)
	n1=0u;
	r_t_it=real_times.begin();
	for (i=0u; i!=n; ++i)
	{
		if (r_t_it==r_t_it_end) break;
		it=ideal_topo[i].second.lower_bound(i+1u);
		it_end=ideal_topo[i].second.end();
		if (i==r_t_it->first)
		{
			it4=r_t_it->second.begin();
			it4_end=r_t_it->second.end();
			for ( ; ; )
			{
				if (it4==it4_end) break;
				if (it->first==it4->first)
				{
					// edge 'i'->'it->first' is coloured
					time_real=it->second-it4->second.first; // temporary
					if (time_real>0.0)
					{
						// red color
						*clr_arr=static_cast<unsigned char>(floor((time_real-gt_min)*gt_max+0.5));
						*bit_arr|=static_cast<quint8>(1u<<n1);
					}
					else
					{
						// blue color
						*clr_arr=static_cast<unsigned char>(floor(0.5-(time_real+lt_min)*lt_max));
						*bit_arr|=static_cast<quint8>(2u<<n1);
					}
					++clr_arr;
					++it4;
				}
				++it;
				n1+=2u;
				if (n1==8u)
				{
					n1=0u;
					++bit_arr;
				}
			}
			++r_t_it;
		}
		// "ignore" color
		for ( ; it!=it_end; ++it,n1+=2u) ;
		bit_arr+=(n1>>3u);
		n1&=0x7;
	}

	real_times.clear(); // no longer necessary

	/* turn adjacency list into adjacency matrix */
	/* fill distance matrix with edges' lengths */
	matr=static_cast<double*>(calloc(n*n,sizeof(double)));
	if (matr==NULL) { delete ideal_topo_view; NOT_ENOUGH_MEM_RET(); }
	edg_cnt=static_cast<unsigned int*>(calloc(n*n,sizeof(int)));
	if (edg_cnt==NULL) { free(matr); delete ideal_topo_view; NOT_ENOUGH_MEM_RET(); }
	ideal_topo_view->edge_counts=edg_cnt;
	dist=matr;
	for (i=0u; i!=n; ++i,edg_cnt+=n,dist+=n)
	{
		for (it=ideal_topo[i].second.begin(),it_end=ideal_topo[i].second.end(); it!=it_end; ++it)
		{
			k=it->first;
			edg_cnt[k]=1u;
			dist[k]=it->second;
		}
	}

	/* assign names */
	ideal_topo_view->host_names=new(std::nothrow) QString[n];
	if (ideal_topo_view->host_names==NULL) { free(matr); delete ideal_topo_view; NOT_ENOUGH_MEM_RET(); }
	for (i=0u; i!=n; ++i)
	{
		ideal_topo_view->host_names[i]=ideal_topo[i].first;
		ideal_topo[i].first.clear();
	}

	ideal_topo.clear(); // no longer necessary

	ideal_topo_view->resize(main_wdg.size());
	ideal_topo_view->show();

	unsigned int edg_num=0u; // number of all edges
	unsigned int edg50_num=0u; // number of edges with length error not less than 50%
	unsigned int edg99_num=0u; // number of edges with length error not less than 99%

	if (!ideal_topo_view->MapGraphInto3D(matr,m_d_impact,edg_num,edg50_num,edg99_num))
	{
		free(matr);
		delete ideal_topo_view;
		NOT_ENOUGH_MEM_RET();
	}
	free(matr);

	ideal_topo_view->setParent(this); // hope that 'ideal_topo_view' will be deleted by Qt
	ideal_topo_view->setAttribute(Qt::WA_DeleteOnClose,true);
	ideal_topo_view->setWindowFlags(ideal_topo_view->windowFlags() | Qt::Window); // tear away from 'this'
	ideal_topo_view->window()->setWindowTitle(QString("PARUS - Network Viewer 2 - ")+
											  tr("\"ideal\" topology for '")+fname_for_tvwidget+"'");
	if (ideal_topo_view->isHidden()) ideal_topo_view->show(); // why does the window disappear sometimes??

	if (edg50_num!=0u)
	{
		const double r_edg_num=100.0/static_cast<double>(edg_num);
		QString mes=tr("Graph building");
		mes+=tr(" was not precise:");
		mes+=QString("<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <b>%1%</b> ")\
					.arg(static_cast<double>(edg50_num)*r_edg_num,0,'f',2);
		mes+=tr("of edges have %1% length error or greater").arg(50); // 'arg(const)' is used to minimize
																	  // the number of translated strings
		if (edg99_num!=0u)
		{
			mes+=QString("<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <b>%1%</b> ")\
						.arg(static_cast<double>(edg99_num)*r_edg_num,0,'f',2);
			mes+=tr("of edges have %1% length error or greater").arg(99);
		}
		(mes+="<br><br>")+=tr("(the graph has <b>%1</b> edges in all)<br>").arg(edg_num);
		emit SendMessToLog(MainWindow::Info,my_sign,mes);
		QMessageBox::information(ideal_topo_view,tr("Graph building"),mes);
	}

	ideal_topo_view->updateGL();
}

void TopologyViewer::keyPressEvent (QKeyEvent *key_event) {
	if (key_event->key()==Qt::Key_Escape) // 'Esc' was pressed -> close this tab
	{
		emit CloseOnEscape(this);
		return;
	}

	if ((key_event->modifiers()==Qt::ControlModifier) && (key_event->key()==Qt::Key_O))
	{
		// 'Ctrl'+'O' was pressed -> open file with "ideal" topology to compare with retrieved topology
		CompareTopologies();
		return;
	}

	main_wdg.keyPressEvent(key_event); // "redirection" of the event
}

void TVWidget::keyPressEvent (QKeyEvent *key_event) {
	static const float shift_xy=0.2f,shift_zz=0.1f;

	switch (key_event->key())
	{
	  case Qt::Key_A: // shift "camera" left
	  	  shift_x-=shift_xy;
	  	  break;
	  case Qt::Key_D: // shift "camera" right
	  	  shift_x+=shift_xy;
	  	  break;
	  case Qt::Key_W: // shift "camera" up
	  	  shift_y-=shift_xy;
	  	  break;
	  case Qt::Key_S: // shift "camera" down
	  	  shift_y+=shift_xy;
	  	  break;
	  case Qt::Key_Z: // shift "camera" backward (objects move farther from observer)
	  	  shift_z+=shift_zz;
	  	  break;
	  case Qt::Key_X: // shift "camera" forward (objects move nearer to observer)
	  	  shift_z-=shift_zz;
	  	  break;
	  /*case Qt::Key_Plus:
	  	  if (renderer->Zoom(INC)) RenderBox();
	  	  break;
	  case Qt::Key_Minus:
	  	  if (renderer->Zoom(DEC)) RenderBox();
	  	  break;*/
	  default: return;
	}
	ApplyTransform();
	updateGL();
}

void TVWidget::mousePressEvent (QMouseEvent *mouse_event) {
	if (mouse_event->button()==Qt::LeftButton)
	{
		mouse_pressed=true;
		click_disabled=false;
		x_move=mouse_event->x();
		y_move=mouse_event->y();
	}
    if (mouse_event->button()==Qt::RightButton)
    {
        QMenu* contextMenu = new QMenu ( this );
        Q_CHECK_PTR ( contextMenu );
          contextMenu->addAction ( "Save" , this , SLOT(SaveImageMenu()) );
          contextMenu->popup( QCursor::pos() );
          contextMenu->exec ();
          delete contextMenu;
          contextMenu = 0;
    }
}

void TVWidget::mouseMoveEvent (QMouseEvent *mouse_event) {
	if (mouse_pressed)
	{
		const int new_x=mouse_event->x(),new_y=mouse_event->y();
		static const int min_move=3;

		if ((new_x<0) || (new_y<0) || (new_x>=width()) || (new_y>=height()) ||
			((new_x<x_move+min_move) && (new_x+min_move>x_move) &&
			 (new_y<y_move+min_move) && (new_y+min_move>y_move))) return;

		click_disabled=true;

		// rotation in OXZ
		alpha+=(static_cast<float>(180*(new_x-x_move))/static_cast<float>(width()));
		// rotation in OYZ
		beta+=(static_cast<float>(180*(y_move-new_y))/static_cast<float>(height()));

		if (alpha>360.0f) alpha-=360.0f;
		else
			if (alpha<-360.0f) alpha+=360.0f;
		if (beta>360.0f) beta-=360.0f;
		else
			if (beta<-360.0f) beta+=360.0f;

		// move vector origin to the new point
		x_move=new_x;
		y_move=new_y;

		ApplyTransform();
		updateGL();
	}
}


void TVWidget::mouseReleaseEvent (QMouseEvent *mouse_event) {
	if (mouse_event->button()==Qt::LeftButton)
	{
		const int m_x=mouse_event->x(),m_y=mouse_event->y();

		mouse_pressed=false;
		if ((m_x<0) || (m_y<0) || (m_x>=width()) || (m_y>=height())) click_disabled=false;
	}
}

void TVWidget::ApplyTransform (void) {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0);

	glTranslatef(shift_x,shift_y,shift_z+geom_c_z);

	glRotatef(alpha,0.0f,1.0f,0.0f);
	glRotatef(beta,1.0f,0.0f,0.0f);

	glTranslatef(0.0f,0.0f,-geom_c_z);
}

void TVWidget::tvCone (const float radius, const float height, const unsigned int slices) {
	if (slices<2u) return;

	const float alpha=(M_PI+M_PI)/static_cast<float>(slices);
	const float sina=sinf(alpha),cosa=cosf(alpha);
	unsigned int i;
	float dir_x=radius,dir_y=0.0f,tmp;

	// cone surface
	glBegin(GL_TRIANGLE_FAN);
	  glNormal3f(0.0f,0.0f,1.0f); // top point has this normal
	  glVertex3f(0.0f,0.0f,height);
	  glNormal3f(1.0f,0.0f,0.0f); // not a real normal!
	  glVertex3f(radius,0.0f,0.0f);
	  for (i=1u; i!=slices; ++i)
	  {
	  	  tmp=dir_x;
	  	  dir_x=tmp*cosa-dir_y*sina;
	  	  dir_y=tmp*sina+dir_y*cosa;
	  	  tmp=sqrtf(1.0f/(dir_x*dir_x+dir_y*dir_y));
	  	  glNormal3f(dir_x*tmp,dir_y*tmp,0.0f); // not a real normal!
	  	  glVertex3f(dir_x,dir_y,0.0f);
	  }
	  glNormal3f(1.0f,0.0f,0.0f); // not a real normal!
	  glVertex3f(radius,0.0f,0.0f);
	glEnd();
	// disk at the bottom
	dir_x=radius;
	dir_y=0.0f;
	glNormal3f(0.0f,0.0f,-1.0f); // for the whole polygon
	glBegin(GL_TRIANGLE_FAN);
	  glVertex3f(0.0f,0.0f,0.0f);
	  glVertex3f(radius,0.0f,0.0f);
	  for (i=1u; i!=slices; ++i)
	  {
	  	  tmp=dir_x;
	  	  dir_x=tmp*cosa-dir_y*sina;
	  	  dir_y=tmp*sina+dir_y*cosa;
	  	  glVertex3f(dir_x,dir_y,0.0f);
	  }
	  glVertex3f(radius,0.0f,0.0f);
	glEnd();
}

void TVWidget::initializeGL () {
	glClearColor(backgr_clr,backgr_clr,backgr_clr,1.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);

	glShadeModel(GL_SMOOTH/*GL_FLAT*/);

	//glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glLineWidth(3.0f);
	glEnable(GL_LINE_SMOOTH);
}

void TVWidget::paintGL () {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (points_x==NULL) return;

	static const float vert_rad=0.025f,edg_rad=0.01f,rad_to_deg=180.0f/M_PI;
	GLfloat mat_clr_diff[]={32.0f/51.0f,0.0f,0.0f};
	static const GLfloat mat_clr_spec[]={0.0f,0.0f,0.0f};
	unsigned int i,j;
	unsigned int *edgs=edge_counts;
	float coef;

	GLUquadric *quadr_obj=gluNewQuadric();

	glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,mat_clr_spec);

	if (i_v_color==NULL)
	{
		// "real" topology

		unsigned int from,to;
		unsigned int *edgs2;
		float rb,len;
		static const unsigned int simplex_times=2u;
		static const float exist_eps=0.5f;
		bool arrow;

		// vertices
		glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mat_clr_diff);//glColor3ub(160u,0u,0u);
		for (i=0u; i!=x_num; ++i)
		{
			glPushMatrix();
			glTranslatef(points_x[i],points_y[i],points_z[i]);
			gluSphere(quadr_obj,vert_rad,4,2);
			glPopMatrix();
		}

		// edges
		for (i=0u; i!=x_num; ++i)
		{
			j=i+1u;
			edgs+=j;
			edgs2=edge_counts+(j*x_num+i);
			for ( ; j!=x_num; ++j,++edgs,edgs2+=x_num)
			{
				from=i;
				to=j;
				if (*edgs<min_edg_count)
				{
					if (*edgs2<min_edg_count)
						continue; // assume no edge
					// a simplex channel j->i
					from=j;
					to=i;
					arrow=true;
				}
				else
				{
					if (*edgs2<min_edg_count)
						// a simplex channel i->j
						arrow=true;
					else
					{
						if (*edgs>*edgs2*simplex_times)
							// a simplex channel i->j
							arrow=true;
						else
						{
							if (*edgs2>*edgs*simplex_times)
							{
								// a simplex channel j->i
								from=j;
								to=i;
								arrow=true;
							}
							else
								// assume a duplex channel
								arrow=false;
						}
					}
				}
				glPushMatrix();
				glTranslatef(points_x[from],points_y[from],points_z[from]);
				len=sqrtf((points_x[from]-points_x[to])*(points_x[from]-points_x[to])+
						  (points_y[from]-points_y[to])*(points_y[from]-points_y[to])+
						  (points_z[from]-points_z[to])*(points_z[from]-points_z[to]));
				glRotatef(acosf((points_z[to]-points_z[from])/len)*rad_to_deg,
						  points_y[from]-points_y[to],points_x[to]-points_x[from],0.0f);
				// magic formula!
				coef=static_cast<float>(static_cast<double>(*edgs**edgs+*edgs2**edgs2)/
										static_cast<double>(z_num*(*edgs+*edgs2)));
				//coef=static_cast<float>(*edgs2)/static_cast<float>(z_num);
				//printf("%f\n",coef);
				rb=backgr_clr*(1.0f-coef);
				if (coef+rb+exist_eps<1.0f)
				{
					glDisable(GL_LIGHTING);
					glColor3f(rb,coef+rb,rb);
				}
				else
				{
					mat_clr_diff[0]=mat_clr_diff[2]=rb;
					mat_clr_diff[1]=coef+rb;
					glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mat_clr_diff);
				}
				gluCylinder(quadr_obj,edg_rad,edg_rad,len,9,1);
				/*glBegin(GL_LINES);
				  glVertex3f(0.0f,0.0f,0.0f);
				  glVertex3f(0.0f,0.0f,len);
				glEnd();*/
				if (arrow)
				{
					coef=len*0.15f;
					coef=(coef>1.0f)? 1.0f : coef;
					glTranslatef(0.0f,0.0f,len-coef+vert_rad);
					tvCone(edg_rad+edg_rad,coef,9u);
				}
				if (!glIsEnabled(GL_LIGHTING)) glEnable(GL_LIGHTING);
				glPopMatrix();
			}
		}

		// vertices' labels
		if (show_host_names)
		{
			//glDisable(GL_DEPTH_TEST);
			glDisable(GL_LIGHTING);
			glColor3fv(mat_clr_spec); // 'mat_clr_spec' consists of zeroes
			coef=vert_rad*1.3f;
			for (i=0u; i!=x_num; ++i)
				renderText(points_x[i]+coef,points_y[i]+coef,points_z[i]+coef,host_names[i]);
			glEnable(GL_LIGHTING);
			//glEnable(GL_DEPTH_TEST);
		}
	}
	else
	{
		// "ideal" topology

		quint8 *bit_arr; // iterator for bit arrays
		quint8 one_byte; // stores '*bit_arr'
		unsigned int n1; // bit counter
		bool gray=true; // flag to minimize calls to glMaterialfv() due to color switching
		unsigned char *clr_arr; // iterator for 'i_e_color_val'
		static const GLfloat _1_div_255=1.0f/255.0f;

		// vertices
		mat_clr_diff[0]=mat_clr_diff[1]=mat_clr_diff[2]=ignore_clr;
		glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mat_clr_diff);
		bit_arr=i_v_color;
		one_byte=*bit_arr;
		++bit_arr;
		n1=0u;
		for (i=0u; i!=x_num; ++i)
		{
			if ((one_byte & 0x1)==0u)
			{
				// "ignore" color
				if (!gray)
				{
					gray=true;
					mat_clr_diff[0]=mat_clr_diff[1]=mat_clr_diff[2]=ignore_clr;
					glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mat_clr_diff);
				}
			}
			else
			{
				// magenta color
				if (gray)
				{
					gray=false;
					mat_clr_diff[0]=mat_clr_diff[2]=1.0f;
					mat_clr_diff[1]=0.0f;
					glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mat_clr_diff);
				}
			}
			++n1;
			if (n1==8u)
			{
				n1=0u;
				one_byte=*bit_arr;
				++bit_arr;
			}
			else
				one_byte>>=1u;
			glPushMatrix();
			glTranslatef(points_x[i],points_y[i],points_z[i]);
			gluSphere(quadr_obj,vert_rad,4,2);
			glPopMatrix();
		}

		// edges
		bit_arr=i_e_color;
		one_byte=*bit_arr;
		++bit_arr;
		clr_arr=i_e_color_val;
		n1=0u;
		for (i=0u; i!=x_num; ++i)
		{
			j=i+1u;
			for (edgs+=j; j!=x_num; ++j,++edgs)
			{
				if (*edgs==0u) continue;
				switch (one_byte & 0x3)
				{
					case 0u: // "ignore" color
						mat_clr_diff[0]=mat_clr_diff[1]=mat_clr_diff[2]=ignore_clr;
						break;
					case 1u: // red color
						mat_clr_diff[0]=static_cast<GLfloat>(*clr_arr)*_1_div_255;
						++clr_arr;
						mat_clr_diff[1]=mat_clr_diff[2]=0.0f;
						break;
					case 2u: // blue color
						mat_clr_diff[2]=static_cast<GLfloat>(*clr_arr)*_1_div_255;
						++clr_arr;
						mat_clr_diff[0]=mat_clr_diff[1]=0.0f;
						break;
				}
				n1+=2u;
				if (n1==8u)
				{
					n1=0u;
					one_byte=*bit_arr;
					++bit_arr;
				}
				else
					one_byte>>=2u;
				glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,mat_clr_diff);
				glPushMatrix();
				glTranslatef(points_x[i],points_y[i],points_z[i]);
				coef=sqrtf((points_x[i]-points_x[j])*(points_x[i]-points_x[j])+
						  (points_y[i]-points_y[j])*(points_y[i]-points_y[j])+
						  (points_z[i]-points_z[j])*(points_z[i]-points_z[j]));
				glRotatef(acosf((points_z[j]-points_z[i])/coef)*rad_to_deg,
						  points_y[i]-points_y[j],points_x[j]-points_x[i],0.0f);
				gluCylinder(quadr_obj,edg_rad,edg_rad,coef,9,1);
				glPopMatrix();
			}
		}

		// vertices' labels
		//glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glColor3fv(mat_clr_spec); // 'mat_clr_spec' consists of zeroes
		coef=vert_rad*1.3f;
		for (i=0u; i!=x_num; ++i)
			renderText(points_x[i]+coef,points_y[i]+coef,points_z[i]+coef,host_names[i]);
		glEnable(GL_LIGHTING);
		//glEnable(GL_DEPTH_TEST);
	}

	gluDeleteQuadric(quadr_obj);
}

void TVWidget::resizeGL (int width, int height) {
	glViewport(0,0,width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,static_cast<double>(width)/static_cast<double>(height),1.0,1000.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0);

	const GLfloat light_pos[]={0.0f,0.0f,0.0f,1.0f};
	glLightfv(GL_LIGHT0,GL_POSITION,light_pos);

	ApplyTransform();
}

TVWidget::~TVWidget () {
	if (edge_counts!=NULL) free(edge_counts);
	if (points_x!=NULL) free(points_x);
	if (points_y!=NULL) free(points_y);
	if (points_z!=NULL) free(points_z);
    if (host_names!=NULL) delete[] host_names;
    if (i_v_color!=NULL) free(i_v_color);
    if (i_e_color!=NULL) free(i_e_color);
    if (i_e_color_val!=NULL) free(i_e_color_val);
}

TopologyViewer::~TopologyViewer () {
	if (ncdf_files!=NULL)
    {
		if (ncdf_files->v_file!=-1) nc_close(ncdf_files->v_file);
		if (ncdf_files->d_file!=-1) nc_close(ncdf_files->d_file);
		free(ncdf_files);
    }
    if (txt_files!=NULL)
    {
		if (txt_files->v_file!=NULL) fclose(txt_files->v_file);
		if (txt_files->d_file!=NULL) fclose(txt_files->d_file);
		free(txt_files);
    }
	if (hor_layout!=NULL) delete hor_layout;
}

