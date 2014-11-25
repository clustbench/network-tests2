/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created: Mon Nov 3 13:37:53 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDockWidget>
#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QStatusBar>
#include <QtGui/QTabWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QGridLayout *gridLayout;
    QTabWidget *tabWidget;
    QWidget *TWT_LoadTab;
    QVBoxLayout *verticalLayout_3;
    QLabel *label;
    QSpacerItem *verticalSpacer;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QFrame *gridFrame;
    QGridLayout *gridLayout_6;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QFrame *verticalFrame;
    QVBoxLayout *verticalLayout_4;
    QRadioButton *f_type_NetCDF;
    QRadioButton *f_type_Text;
    QWidget *dummy;
    QFrame *verticalFrame_2;
    QVBoxLayout *verticalLayout_11;
    QRadioButton *wmSingle;
    QRadioButton *wmWithDev;
    QRadioButton *wmCompare;
    QFrame *verticalFrame_3;
    QVBoxLayout *verticalLayout_12;
    QRadioButton *view2D;
    QRadioButton *view3D;
    QRadioButton *viewTopology;
    QFrame *line_2;
    QFrame *line_3;
    QFrame *line_4;
    QFrame *line_5;
    QFrame *line_6;
    QFrame *verticalFrame_4;
    QVBoxLayout *verticalLayout_6;
    QPushButton *Start;
    QSpacerItem *horizontalSpacer_2;
    QSpacerItem *verticalSpacer_2;
    QMenuBar *menuBar;
    QMenu *menuTools;
    QAction *actionShow_logs;
    QMenu *menuChLang;
    QAction *actionEngLang;
    QAction *actionRusLang;
    QMenu *menuHelp;
    QAction *actionAbout;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents_2;
    QGridLayout *gridLayout_2;
    QTextEdit *TE_Log;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(800, 560);
        MainWindow->setMinimumSize(QSize(800, 550));
        MainWindow->setWindowTitle(QString::fromUtf8("PARUS - Network Viewer Qt v2"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/img/img/logotype.png"), QSize(), QIcon::Normal, QIcon::Off);
        MainWindow->setWindowIcon(icon);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        centralWidget->setEnabled(true);
        gridLayout = new QGridLayout(centralWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(0, 0, 0, 0);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy);
        tabWidget->setDocumentMode(true);
        tabWidget->setTabsClosable(true);
        TWT_LoadTab = new QWidget();
        TWT_LoadTab->setObjectName(QString::fromUtf8("TWT_LoadTab"));
        TWT_LoadTab->setEnabled(true);
        sizePolicy.setHeightForWidth(TWT_LoadTab->sizePolicy().hasHeightForWidth());
        TWT_LoadTab->setSizePolicy(sizePolicy);
        verticalLayout_3 = new QVBoxLayout(TWT_LoadTab);
        verticalLayout_3->setSpacing(0);
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        label = new QLabel(TWT_LoadTab);
        label->setObjectName(QString::fromUtf8("label"));
        label->setEnabled(true);
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy1);
        label->setMinimumSize(QSize(200, 200));
        label->setText(QString::fromUtf8(""));
        label->setPixmap(QPixmap(QString::fromUtf8(":/img/img/logotype.png")));
        label->setScaledContents(false);
        label->setAlignment(Qt::AlignHCenter|Qt::AlignTop);
        label->setTextInteractionFlags(Qt::NoTextInteraction);

        verticalLayout_3->addWidget(label);

        verticalSpacer = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        gridFrame = new QFrame(TWT_LoadTab);
        gridFrame->setObjectName(QString::fromUtf8("gridFrame"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(gridFrame->sizePolicy().hasHeightForWidth());
        gridFrame->setSizePolicy(sizePolicy2);
        gridFrame->setFrameShape(QFrame::Box);
        gridFrame->setFrameShadow(QFrame::Sunken);
        gridLayout_6 = new QGridLayout(gridFrame);
        gridLayout_6->setSpacing(6);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        label_2 = new QLabel(gridFrame);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        label_2->setFont(font);
        label_2->setFrameShape(QFrame::NoFrame);
        label_2->setFrameShadow(QFrame::Sunken);
        label_2->setTextInteractionFlags(Qt::NoTextInteraction);

        gridLayout_6->addWidget(label_2, 1, 0, 1, 1);

        label_3 = new QLabel(gridFrame);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setFont(font);
        label_3->setFrameShape(QFrame::NoFrame);
        label_3->setFrameShadow(QFrame::Sunken);
        label_3->setTextInteractionFlags(Qt::NoTextInteraction);

        gridLayout_6->addWidget(label_3, 1, 2, 1, 1);

        label_4 = new QLabel(gridFrame);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setFont(font);
        label_4->setFrameShape(QFrame::NoFrame);
        label_4->setFrameShadow(QFrame::Sunken);
        label_4->setMidLineWidth(0);
        label_4->setTextInteractionFlags(Qt::NoTextInteraction);

        gridLayout_6->addWidget(label_4, 1, 5, 1, 1);

        verticalFrame = new QFrame(gridFrame);
        verticalFrame->setObjectName(QString::fromUtf8("verticalFrame"));
        verticalFrame->setFrameShape(QFrame::NoFrame);
        verticalFrame->setFrameShadow(QFrame::Sunken);
        verticalLayout_4 = new QVBoxLayout(verticalFrame);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        f_type_NetCDF = new QRadioButton(verticalFrame);
        f_type_NetCDF->setObjectName(QString::fromUtf8("f_type_NetCDF"));
        f_type_NetCDF->setText(QString::fromUtf8("NetCDF"));
        f_type_NetCDF->setChecked(true);
        f_type_NetCDF->setAutoExclusive(true);

        verticalLayout_4->addWidget(f_type_NetCDF);

        f_type_Text = new QRadioButton(verticalFrame);
        f_type_Text->setObjectName(QString::fromUtf8("f_type_Text"));

        verticalLayout_4->addWidget(f_type_Text);

        dummy = new QWidget(verticalFrame);
        dummy->setObjectName(QString::fromUtf8("dummy"));
        dummy->setEnabled(false);

        verticalLayout_4->addWidget(dummy);


        gridLayout_6->addWidget(verticalFrame, 3, 0, 1, 1);

        verticalFrame_2 = new QFrame(gridFrame);
        verticalFrame_2->setObjectName(QString::fromUtf8("verticalFrame_2"));
        verticalFrame_2->setFrameShape(QFrame::NoFrame);
        verticalFrame_2->setFrameShadow(QFrame::Sunken);
        verticalLayout_11 = new QVBoxLayout(verticalFrame_2);
        verticalLayout_11->setSpacing(6);
        verticalLayout_11->setContentsMargins(11, 11, 11, 11);
        verticalLayout_11->setObjectName(QString::fromUtf8("verticalLayout_11"));
        wmSingle = new QRadioButton(verticalFrame_2);
        wmSingle->setObjectName(QString::fromUtf8("wmSingle"));
        wmSingle->setChecked(true);
        wmSingle->setAutoExclusive(true);

        verticalLayout_11->addWidget(wmSingle);

        wmWithDev = new QRadioButton(verticalFrame_2);
        wmWithDev->setObjectName(QString::fromUtf8("wmWithDev"));

        verticalLayout_11->addWidget(wmWithDev);

        wmCompare = new QRadioButton(verticalFrame_2);
        wmCompare->setObjectName(QString::fromUtf8("wmCompare"));

        verticalLayout_11->addWidget(wmCompare);


        gridLayout_6->addWidget(verticalFrame_2, 3, 2, 1, 1);

        verticalFrame_3 = new QFrame(gridFrame);
        verticalFrame_3->setObjectName(QString::fromUtf8("verticalFrame_3"));
        verticalFrame_3->setFrameShape(QFrame::NoFrame);
        verticalFrame_3->setFrameShadow(QFrame::Sunken);
        verticalLayout_12 = new QVBoxLayout(verticalFrame_3);
        verticalLayout_12->setSpacing(6);
        verticalLayout_12->setContentsMargins(11, 11, 11, 11);
        verticalLayout_12->setObjectName(QString::fromUtf8("verticalLayout_12"));
        view2D = new QRadioButton(verticalFrame_3);
        view2D->setObjectName(QString::fromUtf8("view2D"));
        view2D->setText(QString::fromUtf8("2D"));
        view2D->setChecked(true);
        view2D->setAutoExclusive(true);

        verticalLayout_12->addWidget(view2D);

        view3D = new QRadioButton(verticalFrame_3);
        view3D->setObjectName(QString::fromUtf8("view3D"));
        view3D->setText(QString::fromUtf8("3D"));

        verticalLayout_12->addWidget(view3D);

        viewTopology = new QRadioButton(verticalFrame_3);
        viewTopology->setObjectName(QString::fromUtf8("viewTopology"));

        verticalLayout_12->addWidget(viewTopology);


        gridLayout_6->addWidget(verticalFrame_3, 3, 5, 1, 1);

        line_2 = new QFrame(gridFrame);
        line_2->setObjectName(QString::fromUtf8("line_2"));
        line_2->setFrameShape(QFrame::VLine);
        line_2->setFrameShadow(QFrame::Sunken);

        gridLayout_6->addWidget(line_2, 1, 1, 1, 1);

        line_3 = new QFrame(gridFrame);
        line_3->setObjectName(QString::fromUtf8("line_3"));
        line_3->setFrameShape(QFrame::VLine);
        line_3->setFrameShadow(QFrame::Sunken);

        gridLayout_6->addWidget(line_3, 1, 4, 1, 1);

        line_4 = new QFrame(gridFrame);
        line_4->setObjectName(QString::fromUtf8("line_4"));
        line_4->setFrameShape(QFrame::HLine);
        line_4->setFrameShadow(QFrame::Sunken);

        gridLayout_6->addWidget(line_4, 2, 0, 1, 6);

        line_5 = new QFrame(gridFrame);
        line_5->setObjectName(QString::fromUtf8("line_5"));
        line_5->setFrameShape(QFrame::VLine);
        line_5->setFrameShadow(QFrame::Sunken);

        gridLayout_6->addWidget(line_5, 3, 1, 1, 1);

        line_6 = new QFrame(gridFrame);
        line_6->setObjectName(QString::fromUtf8("line_6"));
        line_6->setFrameShape(QFrame::VLine);
        line_6->setFrameShadow(QFrame::Sunken);

        gridLayout_6->addWidget(line_6, 3, 4, 1, 1);


        horizontalLayout->addWidget(gridFrame);

        verticalFrame_4 = new QFrame(TWT_LoadTab);
        verticalFrame_4->setObjectName(QString::fromUtf8("verticalFrame_4"));
        verticalFrame_4->setFrameShape(QFrame::Box);
        verticalFrame_4->setFrameShadow(QFrame::Sunken);
        verticalLayout_6 = new QVBoxLayout(verticalFrame_4);
        verticalLayout_6->setSpacing(6);
        verticalLayout_6->setContentsMargins(11, 11, 11, 11);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        Start = new QPushButton(verticalFrame_4);
        Start->setObjectName(QString::fromUtf8("Start"));
        Start->setMinimumSize(QSize(0, 70));
        QFont font1;
        font1.setPointSize(11);
        font1.setBold(true);
        font1.setWeight(75);
        Start->setFont(font1);

        verticalLayout_6->addWidget(Start);


        horizontalLayout->addWidget(verticalFrame_4);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        verticalLayout_3->addLayout(horizontalLayout);

        verticalSpacer_2 = new QSpacerItem(20, 0, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer_2);

        tabWidget->addTab(TWT_LoadTab, icon, QString());

        gridLayout->addWidget(tabWidget, 0, 0, 1, 1);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 800, 23));
        menuTools = new QMenu(menuBar);
        menuTools->setObjectName(QString::fromUtf8("menuTools"));
        actionShow_logs = new QAction(menuTools);
        actionShow_logs->setObjectName(QString::fromUtf8("actionShow_logs"));
        actionShow_logs->setCheckable(true);
        actionShow_logs->setChecked(true);
        actionShow_logs->setSoftKeyRole(QAction::NoSoftKey);
        menuChLang = new QMenu(menuTools);
        menuChLang->setObjectName(QString::fromUtf8("menuChLang"));
        actionEngLang = new QAction(menuChLang);
        actionEngLang->setObjectName(QString::fromUtf8("actionEngLang"));
        actionEngLang->setCheckable(true);
        actionEngLang->setChecked(false);
        actionEngLang->setText(QString::fromUtf8("English"));
        actionEngLang->setAutoRepeat(false);
        actionRusLang = new QAction(menuChLang);
        actionRusLang->setObjectName(QString::fromUtf8("actionRusLang"));
        actionRusLang->setCheckable(true);
        actionRusLang->setChecked(true);
        actionRusLang->setText(QString::fromUtf8("\320\240\321\203\321\201\321\201\320\272\320\270\320\271"));
        actionRusLang->setAutoRepeat(false);
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
        actionAbout = new QAction(menuHelp);
        actionAbout->setObjectName(QString::fromUtf8("actionAbout"));
        MainWindow->setMenuBar(menuBar);
        dockWidget = new QDockWidget(MainWindow);
        dockWidget->setObjectName(QString::fromUtf8("dockWidget"));
        dockWidget->setEnabled(true);
        dockWidget->setMinimumSize(QSize(75, 104));
        dockWidget->setMouseTracking(false);
        dockWidget->setFloating(false);
        dockWidget->setFeatures(QDockWidget::DockWidgetMovable);
        dockWidget->setAllowedAreas(Qt::AllDockWidgetAreas);
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QString::fromUtf8("dockWidgetContents_2"));
        gridLayout_2 = new QGridLayout(dockWidgetContents_2);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        TE_Log = new QTextEdit(dockWidgetContents_2);
        TE_Log->setObjectName(QString::fromUtf8("TE_Log"));
        TE_Log->setLineWrapMode(QTextEdit::NoWrap);
        TE_Log->setReadOnly(true);

        gridLayout_2->addWidget(TE_Log, 0, 0, 1, 1);

        dockWidget->setWidget(dockWidgetContents_2);
        MainWindow->addDockWidget(static_cast<Qt::DockWidgetArea>(8), dockWidget);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuTools->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuTools->addAction(actionShow_logs);
        menuTools->addSeparator();
        menuTools->addAction(menuChLang->menuAction());
        menuChLang->addAction(actionEngLang);
        menuChLang->addAction(actionRusLang);
        menuHelp->addAction(actionAbout);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        label_2->setText(QApplication::translate("MainWindow", "File type", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MainWindow", "Working mode", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MainWindow", "View", 0, QApplication::UnicodeUTF8));
        f_type_Text->setText(QApplication::translate("MainWindow", "Text", 0, QApplication::UnicodeUTF8));
        wmSingle->setText(QApplication::translate("MainWindow", "Single", 0, QApplication::UnicodeUTF8));
        wmWithDev->setText(QApplication::translate("MainWindow", "With deviations", 0, QApplication::UnicodeUTF8));
        wmCompare->setText(QApplication::translate("MainWindow", "Compare 2 files", 0, QApplication::UnicodeUTF8));
        viewTopology->setText(QApplication::translate("MainWindow", "Topology", 0, QApplication::UnicodeUTF8));
        Start->setText(QApplication::translate("MainWindow", "Start", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(TWT_LoadTab), QApplication::translate("MainWindow", "Load", 0, QApplication::UnicodeUTF8));
        menuTools->setTitle(QApplication::translate("MainWindow", "Tools", 0, QApplication::UnicodeUTF8));
        actionShow_logs->setText(QApplication::translate("MainWindow", "Show logs", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        actionShow_logs->setToolTip(QApplication::translate("MainWindow", "Show/hide logs", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        actionShow_logs->setShortcut(QApplication::translate("MainWindow", "Ctrl+L", 0, QApplication::UnicodeUTF8));
        menuChLang->setTitle(QApplication::translate("MainWindow", "Language", 0, QApplication::UnicodeUTF8));
        menuHelp->setTitle(QApplication::translate("MainWindow", "Help", 0, QApplication::UnicodeUTF8));
        actionAbout->setText(QApplication::translate("MainWindow", "About...", 0, QApplication::UnicodeUTF8));
        dockWidget->setWindowTitle(QApplication::translate("MainWindow", "Log", 0, QApplication::UnicodeUTF8));
        Q_UNUSED(MainWindow);
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
