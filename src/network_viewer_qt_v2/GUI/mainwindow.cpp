#include "mainwindow.h"
#include "core/data_netcdf.h"
#include "core/data_text.h"
#include "core/cntrlr_single.h"
#include "core/cntrlr_deviat.h"
#include "core/cntrlr_compare.h"
#include "tabviewer.h"
#include "fullviewer.h"
#include "topoviewer.h"
#include <QMessageBox>
#include <QFileDialog>
#include <QTextCodec>
#include <QLibraryInfo>
#include <QCoreApplication>
#include "err_msgs.h"

QString ErrMsgs::err_msgs[NV::LastCode]; // see declaration in 'err_msgs.h'

void MainWindow::Init (void) {
	setAttribute(Qt::WA_QuitOnClose,true);

	QTextCodec::setCodecForTr(QTextCodec::codecForName(
#ifdef Q_WS_WIN
	"Windows-1251"
#else
	"UTF-8"
#endif
	));
	
	try {
		ui=new Ui::MainWindow;
		ui->setupUi(this);
		app_tr=new QTranslator;
		qt_tr=new QTranslator;
	}
	catch (...) { return; }
	
	app_tr->load("translations/nv_tr_ru.qm");
	qt_tr->load("qt_ru",QLibraryInfo::location(QLibraryInfo::TranslationsPath));
	qApp->installTranslator(qt_tr);
	qApp->installTranslator(app_tr);

	connect(ui->tabWidget,SIGNAL(currentChanged(int)),this,SLOT(ToggleWidgetsIn_2D_3D(const int)));
	connect(ui->tabWidget,SIGNAL(tabCloseRequested(int)),this,SLOT(CloseTab(const int)));
	connect(ui->actionAbout,SIGNAL(triggered()),this,SLOT(ShowAbout()));
	connect(ui->Start,SIGNAL(clicked()),this,SLOT(Load()));
	connect(ui->actionShow_logs,SIGNAL(toggled(bool)),this,SLOT(ToggleLog(const bool)));
	connect(ui->actionEngLang,SIGNAL(toggled(bool)),this,SLOT(SwitchLanguageToEng(const bool)));
	connect(ui->actionRusLang,SIGNAL(toggled(bool)),this,SLOT(SwitchLanguageToRus(const bool)));
	// "compare 2 files" mode is not compatible with topology viewer
	connect(ui->wmCompare,SIGNAL(toggled(bool)),ui->viewTopology,SLOT(setDisabled(bool)));
	connect(ui->viewTopology,SIGNAL(toggled(bool)),ui->wmCompare,SLOT(setDisabled(bool)));
	
	const QString welcome(tr("Welcome to Network Viewer Qt v2!"));
	QString to_log("<p align=\"center\"><b>");
	(to_log+=welcome)+="</b></p><p></p>";
	ui->TE_Log->append(to_log);
	this->statusBar()->showMessage(welcome,7000); // show during 7 seconds
	puts(welcome.toLocal8Bit().constData());
}

MainWindow::~MainWindow () {
	for (int i=ui->tabWidget->count()-1; i>0; --i)
	{
		delete ui->tabWidget->widget(i);
		ui->tabWidget->removeTab(i);
	}
	delete ui;
	qApp->removeTranslator(app_tr);
	qApp->removeTranslator(qt_tr);
	delete qt_tr;
	delete app_tr;
}

void MainWindow::ShowAbout (void) {
	QMessageBox::about(this,tr("About PARUS Network Viewer Qt v2"),
					   tr("<p>Network Viewer Qt v2 is a part of PARUS project.<br>"
					   	  "Link: <a href=\"http://parus.sf.net/\">http://parus.sf.net/</a><br>"
					   	  "<br><u><b>Authors:</b></u><br><br>"
					   	  "<i>Salnikov A.N.</i>&nbsp; &nbsp; &nbsp;salnikov@cmc.msu.ru<br>"
					   	  "<i>Andreev D.Y.</i><br>"
					   	  "<i>Lebedev R.D.</i>&nbsp; &nbsp; &nbsp; rmn.lebedev@gmail.com<br>"
					   	  "<i>Bannikov P.S.</i>&nbsp; &nbsp; &nbsp;pashokkk@bk.ru<br>"
					   	  "<br>Copyright &copy; 2013&nbsp; &nbsp; Pavel S. Bannikov</p>"));
}

void MainWindow::ChangeTabTitle (QWidget *wdg, const QString &title) {
	ui->tabWidget->setTabText(ui->tabWidget->indexOf(wdg),title);
}

void MainWindow::CloseTab (const int index) {
	if (index>0)
	{
		delete ui->tabWidget->widget(index);
		ui->tabWidget->currentWidget()->setFocus(); // without this line widgets ignored keyboard
	}
}

void MainWindow::CloseTab (QWidget *wdg) { CloseTab(ui->tabWidget->indexOf(wdg)); }

void MainWindow::AddMsgToLog (const MsgType type, const QString &sign, const QString &msg0) {
	if (msg0.isEmpty()) return;
	
	QString msg("<p align=\"left\">");
	switch (type)
	{
		case Error: (((msg+="<span style=\"color:red\">[")+=sign)+="] <b>")+=tr("E:"); break;
		case Success: (((msg+="<span style=\"color:green\">[")+=sign)+="] <b>")+=tr("S:"); break;
		case Info: (((msg+="<span style=\"color:olive\">[")+=sign)+="] <b>")+=tr("I:"); break;
	}
	((msg+="</b></span> ")+=msg0)+="</p>";
	
	ui->TE_Log->append(msg);
	
	QTextDocument doc; // for converting 'msg' to plain text
	doc.setHtml(msg);
	const QString plain_msg=doc.toPlainText();
	puts(plain_msg.toLocal8Bit().constData());
	
	this->statusBar()->showMessage(plain_msg,7000); // show during 7 seconds
}

void MainWindow::Load (void) {
	QString data_fname(QFileDialog::getOpenFileName(this,ui->wmCompare->isChecked()? tr("Open first data file") :
																					 tr("Open data file")));
	// immediate processing of all paint events and such
	while (QCoreApplication::hasPendingEvents())
		QCoreApplication::processEvents();

	QString deviat_fname,hosts_fname;
	FILE *file;
	int working_mode;

	if (data_fname.isEmpty()) return;
	file=fopen(data_fname.toLocal8Bit().constData(),"r");
	if (file==NULL)
	{
		AddMsgToLog(Error,"Main",ErrMsgs::ToString(NV::CannotOpen,1,&data_fname));
		return;
	}
	fclose(file);

	if (ui->wmSingle->isChecked()) // "single"
		working_mode=0;
	else
	{
		if (ui->wmWithDev->isChecked()) // "with deviations"
		{
			working_mode=1;
			deviat_fname=QFileDialog::getOpenFileName(this,tr("Open file with deviations"));
		}
		else // "compare 2 files" (we cannot be here if 'ui->viewTopology' is checked)
		{
			working_mode=2;
			// 'deviat_filename' here means the second file
			deviat_fname=QFileDialog::getOpenFileName(this,tr("Open second data file"));
		}
		
		// immediate processing of all paint events and such
		while (QCoreApplication::hasPendingEvents())
			QCoreApplication::processEvents();

		if (data_fname==deviat_fname)
		{
			AddMsgToLog(Error,"Main",ErrMsgs::ToString(NV::SameFileTwice));
			return;
		}
		file=fopen(deviat_fname.toLocal8Bit().constData(),"r");
		if (file==NULL)
		{
			AddMsgToLog(Error,"Main",ErrMsgs::ToString(NV::CannotOpen,1,&deviat_fname));
			return;
		}
		fclose(file);
	}

	int ind_of_slash=data_fname.lastIndexOf('/');
	if (ind_of_slash<0) ind_of_slash=data_fname.lastIndexOf('\\');
	
	// save file type to ensure equal types in case of 2 files
	const IData::Type f_type=ui->f_type_NetCDF->isChecked()? IData::NetCDF : IData::Txt;
	
	if (f_type==IData::NetCDF)
	{
		const int ind_of_undl=data_fname.lastIndexOf('_');

		if ((ind_of_undl>=0) && (ind_of_undl>ind_of_slash))
		{
			(hosts_fname=data_fname.left(ind_of_undl))+="_hosts.txt";
			file=fopen(hosts_fname.toLocal8Bit().constData(),"r"); // may return NULL
		}
		else file=NULL;
		if (file==NULL)
		{
			/* insert spaces into 'data_fname' and 'hosts_fname' 
			   to split long file names in message box */
			QString data_fn(data_fname);
			if (data_fname.indexOf('/')>=0)
			{
				data_fn.replace('/'," / ");
				hosts_fname.replace('/'," / ");
			}
			else
			{
				data_fn.replace('\\'," \\ ");
				hosts_fname.replace('\\'," \\ ");
			}

			if (QMessageBox::warning(this,tr("Warning"),
									 tr("Failed to automatically find file with hosts' names for the file<br><b>"
									 	"'%1 '</b><br>(<i>Assumed</i> <b>'%2 '</b>).<br><br>Do you want to locate"
									 	" it manually?").arg(data_fn).arg(hosts_fname),
									 QMessageBox::Yes | QMessageBox::No)==QMessageBox::Yes)
			{
				hosts_fname=QFileDialog::getOpenFileName(this,tr("Open file with hosts"),
														 data_fname.left(ind_of_slash));
				// immediate processing of all paint events and such
				while (QCoreApplication::hasPendingEvents())
					QCoreApplication::processEvents();
			}
			else
				hosts_fname.clear();
		}
		else fclose(file);
	}

	QWidget *new_tab=NULL;
	QString tab_name("..."); // 'tab_name' will look like ".../file_name.ext"
	tab_name+=data_fname.right(data_fname.length()-ind_of_slash);

	if (ui->view2D->isChecked())
	{
		// open 2D viewer (TabViewer)

		IData *data_file1,*data_file2=NULL;
		NV::ErrCode err=NV::Success;
		ICntrlr *cntrlr=NULL;

		try {
			if (f_type==IData::NetCDF)
				data_file1=new Data_NetCDF(data_fname,hosts_fname,err);
			else // txt
				data_file1=new Data_Text(data_fname,err);
		}
		catch (...) {
			// Hmmm, there is not enough memory, but we want to show a message, 
			// that is to allocate memory! So we have to hope that everything will be OK:)
			AddMsgToLog(Error,"2D",ErrMsgs::ToString(NV::NoMem));
			return;
		}
		if (err!=NV::Success)
		{
			delete data_file1;
			AddMsgToLog(Error,"2D",ErrMsgs::ToString(err,1,(err==NV::NoHosts)? &hosts_fname : &data_fname));
			return;
		}

		QString msg=tr("file \"");
		(msg+=data_fname)+=tr("\" is loaded");
		AddMsgToLog(Success,"2D",msg);
		
		if (working_mode==0)
		{
			try {
				cntrlr=new CntrlrSingle(data_file1,err); // "single"
			}
			catch (...) {
				delete data_file1;
				AddMsgToLog(Error,"2D",ErrMsgs::ToString(NV::NoMem));
				return;
			}
		}
		else
		{
			// "with deviations" or "compare 2 files"
			
			try {
				if (f_type==IData::NetCDF)
					data_file2=new Data_NetCDF(deviat_fname,hosts_fname,err);
				else // txt
					data_file2=new Data_Text(deviat_fname,err);
			}
			catch (...) {
				delete data_file1;
				AddMsgToLog(Error,"2D",ErrMsgs::ToString(NV::NoMem));
				return;
			}
			if (err!=NV::Success)
			{
				delete data_file1;
				delete data_file2;
				AddMsgToLog(Error,"2D",ErrMsgs::ToString(err,1,(err==NV::NoHosts)? &hosts_fname : &deviat_fname));
				return;
			}
			
			QString msg=tr("file \"");
			(msg+=deviat_fname)+=tr("\" is loaded");
			AddMsgToLog(Success,"2D",msg);
			
			try {
				if (working_mode==1) // "with deviations"
					cntrlr=new CntrlrDeviation(data_file1,data_file2,err);
				else // "compare 2 files"
					cntrlr=new CntrlrComparison(data_file1,data_file2,err);
			}
			catch (...) {
				delete data_file1;
				delete data_file2;
				AddMsgToLog(Error,"2D",ErrMsgs::ToString(NV::NoMem));
				return;
			}
		}
		if (err!=NV::Success)
		{
			delete cntrlr;
			AddMsgToLog(Error,"2D",ErrMsgs::ToString(err));
			return;
		}

		new_tab=static_cast<QWidget*>(TabViewer::Create(cntrlr,this));
	}
	else
	{
		if (ui->view3D->isChecked())
			// open 3D viewer (FullViewer)
			new_tab=static_cast<QWidget*>(FullViewer::Create(this,f_type,working_mode,data_fname,
															 deviat_fname,hosts_fname,tab_name));
		else
			// open topology viewer
			new_tab=static_cast<QWidget*>(TopologyViewer::Create(this,working_mode!=0,f_type,data_fname,
																 deviat_fname,hosts_fname));
	}

	if (new_tab==NULL)
		AddMsgToLog(Error,"Main",ErrMsgs::ToString(NV::NoViewer));
	else
		ui->tabWidget->setCurrentIndex(ui->tabWidget->addTab(new_tab,tab_name));
}

void MainWindow::ToggleLog (const bool show) {
	ui->dockWidget->setVisible(show && ((ui->tabWidget->currentIndex()==0) || 
										(dynamic_cast<TabViewer*>(ui->tabWidget->currentWidget())!=NULL)));
}

void MainWindow::ToggleWidgetsIn_2D_3D (const int ind) {
	if ((ind!=0) && (dynamic_cast<TabViewer*>(ui->tabWidget->widget(ind))==NULL))
		// because FullViewer and TopologyViewer (not castable to TabViewer) consume the whole window
		ui->dockWidget->hide();
	else
		// 2D viewer (that is TabViewer)
		ui->dockWidget->setVisible(ui->actionShow_logs->isChecked()); // toggle log widget
}

void MainWindow::SwitchLanguageToEng (const bool checked) {
	if (checked)
	{
		ui->actionRusLang->setChecked(false);

		qApp->removeTranslator(app_tr);
		qApp->removeTranslator(qt_tr);
	}
	else
	{
		if (!ui->actionRusLang->isChecked())
			ui->actionEngLang->setChecked(true);
	}
}

void MainWindow::SwitchLanguageToRus (const bool checked) {
	if (checked)
	{
		ui->actionEngLang->setChecked(false);

		qApp->installTranslator(qt_tr);
		qApp->installTranslator(app_tr);
	}
	else
	{
		if (!ui->actionEngLang->isChecked())
			ui->actionRusLang->setChecked(true);
	}
}

