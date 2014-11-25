/********************************************************************************
** Form generated from reading UI file 'tabviewer.ui'
**
** Created: Mon Nov 3 13:37:53 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TABVIEWER_H
#define UI_TABVIEWER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDockWidget>
#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMdiArea>
#include <QtGui/QProgressBar>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QTabWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "qwt_counter.h"
#include "qwt_slider.h"

QT_BEGIN_NAMESPACE

class Ui_ui_TabViewer
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QHBoxLayout *horizontalLayout;
    QwtCounter *SB_MatrixNumber;
    QwtSlider *S_WindowNumber;
    QPushButton *B_Refresh;
    QPushButton *B_Info;
    QMdiArea *mdiArea;
    QProgressBar *progressBar;
    QDockWidget *DW_Info;
    QWidget *dockWidgetContents;
    QVBoxLayout *verticalLayout_5;
    QTabWidget *tabWidget;
    QWidget *DW_Tab_ToDo;
    QVBoxLayout *Lay_DW_Tab_ToDo;
    QGroupBox *GB_DW_Tab_ToDo_State;
    QGridLayout *gridLayout_3;
    QLabel *label_3;
    QLabel *label_5;
    QLabel *L_StateWinFrom;
    QLabel *label_7;
    QLabel *L_StateWinTo;
    QLabel *L_StateWinStatus;
    QGroupBox *GB_DW_Tab_ToDo_LoadWin;
    QGridLayout *gridLayout_6;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *verticalLayout_2;
    QLabel *label;
    QLabel *label_2;
    QVBoxLayout *verticalLayout_3;
    QwtCounter *SB_LoadWinFrom;
    QwtCounter *SB_LoadWinTo;
    QPushButton *B_LoadWindow;
    QPushButton *B_dropWindow;
    QGroupBox *GB_DW_Tab_ToDo_Show;
    QVBoxLayout *verticalLayout_4;
    QPushButton *B_ShowMatrix;
    QGroupBox *GB_showForWin;
    QGridLayout *gridLayout;
    QCheckBox *CB_updFromCurrMtr;
    QSpinBox *SB_row;
    QPushButton *B_ShowRow;
    QSpinBox *SB_column;
    QPushButton *B_ShowColumn;
    QPushButton *B_ShowPair;
    QSpacerItem *verticalSpacer;
    QWidget *DW_Tab_Data;
    QGridLayout *gridLayout_4;
    QTextEdit *TE_InfoData;
    QWidget *DW_Tab_Hosts;
    QGridLayout *gridLayout_5;
    QTextEdit *LW_InfoHosts;

    void setupUi(QMainWindow *TabViewer)
    {
        if (TabViewer->objectName().isEmpty())
            TabViewer->setObjectName(QString::fromUtf8("TabViewer"));
        TabViewer->resize(800, 600);
        centralwidget = new QWidget(TabViewer);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frame = new QFrame(centralwidget);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        SB_MatrixNumber = new QwtCounter(frame);
        SB_MatrixNumber->setObjectName(QString::fromUtf8("SB_MatrixNumber"));
        SB_MatrixNumber->setMinimumSize(QSize(150, 0));
        SB_MatrixNumber->setMaximumSize(QSize(150, 16777215));
        SB_MatrixNumber->setProperty("basicstep", QVariant(1));
        SB_MatrixNumber->setMinValue(0);
        SB_MatrixNumber->setMaxValue(100);

        horizontalLayout->addWidget(SB_MatrixNumber);

        S_WindowNumber = new QwtSlider(frame);
        S_WindowNumber->setObjectName(QString::fromUtf8("S_WindowNumber"));
        S_WindowNumber->setScalePosition(QwtSlider::TopScale);
        S_WindowNumber->setThumbLength(16);

        horizontalLayout->addWidget(S_WindowNumber);

        B_Refresh = new QPushButton(frame);
        B_Refresh->setObjectName(QString::fromUtf8("B_Refresh"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(B_Refresh->sizePolicy().hasHeightForWidth());
        B_Refresh->setSizePolicy(sizePolicy);
        B_Refresh->setMinimumSize(QSize(30, 30));
        B_Refresh->setMaximumSize(QSize(30, 30));
        QFont font;
        font.setFamily(QString::fromUtf8("Times New Roman"));
        font.setPointSize(14);
        font.setBold(true);
        font.setItalic(true);
        font.setWeight(75);
        B_Refresh->setFont(font);
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/img/img/refresh.png"), QSize(), QIcon::Normal, QIcon::Off);
        B_Refresh->setIcon(icon);
        B_Refresh->setIconSize(QSize(20, 20));
        B_Refresh->setAutoDefault(false);
        B_Refresh->setDefault(false);
        B_Refresh->setFlat(false);

        horizontalLayout->addWidget(B_Refresh);

        B_Info = new QPushButton(frame);
        B_Info->setObjectName(QString::fromUtf8("B_Info"));
        B_Info->setMinimumSize(QSize(30, 30));
        B_Info->setMaximumSize(QSize(30, 30));
        QFont font1;
        font1.setFamily(QString::fromUtf8("Times New Roman"));
        font1.setPointSize(20);
        font1.setBold(true);
        font1.setItalic(true);
        font1.setWeight(75);
        B_Info->setFont(font1);
        B_Info->setCheckable(true);
        B_Info->setChecked(true);

        horizontalLayout->addWidget(B_Info);


        gridLayout_2->addLayout(horizontalLayout, 0, 0, 1, 2);


        verticalLayout->addWidget(frame);

        mdiArea = new QMdiArea(centralwidget);
        mdiArea->setObjectName(QString::fromUtf8("mdiArea"));
        QBrush brush(QColor(125, 124, 123, 255));
        brush.setStyle(Qt::SolidPattern);
        mdiArea->setBackground(brush);
        mdiArea->setDocumentMode(false);

        verticalLayout->addWidget(mdiArea);

        progressBar = new QProgressBar(centralwidget);
        progressBar->setObjectName(QString::fromUtf8("progressBar"));
        progressBar->setValue(0);

        verticalLayout->addWidget(progressBar);

        TabViewer->setCentralWidget(centralwidget);
        DW_Info = new QDockWidget(TabViewer);
        DW_Info->setObjectName(QString::fromUtf8("DW_Info"));
        QSizePolicy sizePolicy1(QSizePolicy::Ignored, QSizePolicy::Ignored);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(DW_Info->sizePolicy().hasHeightForWidth());
        DW_Info->setSizePolicy(sizePolicy1);
        DW_Info->setMinimumSize(QSize(292, 498));
        DW_Info->setFloating(false);
        DW_Info->setFeatures(QDockWidget::DockWidgetMovable);
        DW_Info->setAllowedAreas(Qt::LeftDockWidgetArea|Qt::RightDockWidgetArea);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QString::fromUtf8("dockWidgetContents"));
        verticalLayout_5 = new QVBoxLayout(dockWidgetContents);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        tabWidget = new QTabWidget(dockWidgetContents);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy2);
        tabWidget->setDocumentMode(true);
        DW_Tab_ToDo = new QWidget();
        DW_Tab_ToDo->setObjectName(QString::fromUtf8("DW_Tab_ToDo"));
        Lay_DW_Tab_ToDo = new QVBoxLayout(DW_Tab_ToDo);
        Lay_DW_Tab_ToDo->setObjectName(QString::fromUtf8("Lay_DW_Tab_ToDo"));
        GB_DW_Tab_ToDo_State = new QGroupBox(DW_Tab_ToDo);
        GB_DW_Tab_ToDo_State->setObjectName(QString::fromUtf8("GB_DW_Tab_ToDo_State"));
        gridLayout_3 = new QGridLayout(GB_DW_Tab_ToDo_State);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        label_3 = new QLabel(GB_DW_Tab_ToDo_State);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_3->addWidget(label_3, 0, 0, 1, 1);

        label_5 = new QLabel(GB_DW_Tab_ToDo_State);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_3->addWidget(label_5, 1, 0, 1, 1);

        L_StateWinFrom = new QLabel(GB_DW_Tab_ToDo_State);
        L_StateWinFrom->setObjectName(QString::fromUtf8("L_StateWinFrom"));

        gridLayout_3->addWidget(L_StateWinFrom, 1, 1, 1, 1);

        label_7 = new QLabel(GB_DW_Tab_ToDo_State);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_3->addWidget(label_7, 1, 2, 1, 1);

        L_StateWinTo = new QLabel(GB_DW_Tab_ToDo_State);
        L_StateWinTo->setObjectName(QString::fromUtf8("L_StateWinTo"));

        gridLayout_3->addWidget(L_StateWinTo, 1, 3, 1, 1);

        L_StateWinStatus = new QLabel(GB_DW_Tab_ToDo_State);
        L_StateWinStatus->setObjectName(QString::fromUtf8("L_StateWinStatus"));

        gridLayout_3->addWidget(L_StateWinStatus, 0, 1, 1, 1);


        Lay_DW_Tab_ToDo->addWidget(GB_DW_Tab_ToDo_State);

        GB_DW_Tab_ToDo_LoadWin = new QGroupBox(DW_Tab_ToDo);
        GB_DW_Tab_ToDo_LoadWin->setObjectName(QString::fromUtf8("GB_DW_Tab_ToDo_LoadWin"));
        sizePolicy2.setHeightForWidth(GB_DW_Tab_ToDo_LoadWin->sizePolicy().hasHeightForWidth());
        GB_DW_Tab_ToDo_LoadWin->setSizePolicy(sizePolicy2);
        GB_DW_Tab_ToDo_LoadWin->setMinimumSize(QSize(0, 0));
        GB_DW_Tab_ToDo_LoadWin->setMaximumSize(QSize(16777215, 16777215));
        gridLayout_6 = new QGridLayout(GB_DW_Tab_ToDo_LoadWin);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        verticalLayout_2 = new QVBoxLayout();
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        label = new QLabel(GB_DW_Tab_ToDo_LoadWin);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout_2->addWidget(label);

        label_2 = new QLabel(GB_DW_Tab_ToDo_LoadWin);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        verticalLayout_2->addWidget(label_2);


        horizontalLayout_2->addLayout(verticalLayout_2);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        SB_LoadWinFrom = new QwtCounter(GB_DW_Tab_ToDo_LoadWin);
        SB_LoadWinFrom->setObjectName(QString::fromUtf8("SB_LoadWinFrom"));
        SB_LoadWinFrom->setMinimumSize(QSize(150, 0));
        SB_LoadWinFrom->setMaximumSize(QSize(150, 16777215));
        SB_LoadWinFrom->setProperty("basicstep", QVariant(1));
        SB_LoadWinFrom->setMinValue(0);
        SB_LoadWinFrom->setMaxValue(100);

        verticalLayout_3->addWidget(SB_LoadWinFrom);

        SB_LoadWinTo = new QwtCounter(GB_DW_Tab_ToDo_LoadWin);
        SB_LoadWinTo->setObjectName(QString::fromUtf8("SB_LoadWinTo"));
        SB_LoadWinTo->setMinimumSize(QSize(150, 0));
        SB_LoadWinTo->setMaximumSize(QSize(150, 16777215));
        SB_LoadWinTo->setProperty("basicstep", QVariant(1));
        SB_LoadWinTo->setMinValue(0);
        SB_LoadWinTo->setMaxValue(100);

        verticalLayout_3->addWidget(SB_LoadWinTo);


        horizontalLayout_2->addLayout(verticalLayout_3);


        gridLayout_6->addLayout(horizontalLayout_2, 0, 0, 2, 1);

        B_LoadWindow = new QPushButton(GB_DW_Tab_ToDo_LoadWin);
        B_LoadWindow->setObjectName(QString::fromUtf8("B_LoadWindow"));
        sizePolicy2.setHeightForWidth(B_LoadWindow->sizePolicy().hasHeightForWidth());
        B_LoadWindow->setSizePolicy(sizePolicy2);

        gridLayout_6->addWidget(B_LoadWindow, 0, 1, 1, 1);

        B_dropWindow = new QPushButton(GB_DW_Tab_ToDo_LoadWin);
        B_dropWindow->setObjectName(QString::fromUtf8("B_dropWindow"));

        gridLayout_6->addWidget(B_dropWindow, 1, 1, 1, 1);


        Lay_DW_Tab_ToDo->addWidget(GB_DW_Tab_ToDo_LoadWin);

        GB_DW_Tab_ToDo_Show = new QGroupBox(DW_Tab_ToDo);
        GB_DW_Tab_ToDo_Show->setObjectName(QString::fromUtf8("GB_DW_Tab_ToDo_Show"));
        verticalLayout_4 = new QVBoxLayout(GB_DW_Tab_ToDo_Show);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        B_ShowMatrix = new QPushButton(GB_DW_Tab_ToDo_Show);
        B_ShowMatrix->setObjectName(QString::fromUtf8("B_ShowMatrix"));

        verticalLayout_4->addWidget(B_ShowMatrix);

        GB_showForWin = new QGroupBox(GB_DW_Tab_ToDo_Show);
        GB_showForWin->setObjectName(QString::fromUtf8("GB_showForWin"));
        GB_showForWin->setEnabled(true);
        gridLayout = new QGridLayout(GB_showForWin);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        CB_updFromCurrMtr = new QCheckBox(GB_showForWin);
        CB_updFromCurrMtr->setObjectName(QString::fromUtf8("CB_updFromCurrMtr"));
        CB_updFromCurrMtr->setEnabled(true);

        gridLayout->addWidget(CB_updFromCurrMtr, 0, 0, 1, 2);

        SB_row = new QSpinBox(GB_showForWin);
        SB_row->setObjectName(QString::fromUtf8("SB_row"));
        SB_row->setEnabled(true);

        gridLayout->addWidget(SB_row, 1, 0, 1, 1);

        B_ShowRow = new QPushButton(GB_showForWin);
        B_ShowRow->setObjectName(QString::fromUtf8("B_ShowRow"));
        B_ShowRow->setEnabled(true);

        gridLayout->addWidget(B_ShowRow, 1, 1, 1, 1);

        SB_column = new QSpinBox(GB_showForWin);
        SB_column->setObjectName(QString::fromUtf8("SB_column"));
        SB_column->setEnabled(true);

        gridLayout->addWidget(SB_column, 2, 0, 1, 1);

        B_ShowColumn = new QPushButton(GB_showForWin);
        B_ShowColumn->setObjectName(QString::fromUtf8("B_ShowColumn"));
        B_ShowColumn->setEnabled(true);

        gridLayout->addWidget(B_ShowColumn, 2, 1, 1, 1);

        B_ShowPair = new QPushButton(GB_showForWin);
        B_ShowPair->setObjectName(QString::fromUtf8("B_ShowPair"));
        B_ShowPair->setEnabled(true);

        gridLayout->addWidget(B_ShowPair, 3, 0, 1, 2);


        verticalLayout_4->addWidget(GB_showForWin);


        Lay_DW_Tab_ToDo->addWidget(GB_DW_Tab_ToDo_Show);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        Lay_DW_Tab_ToDo->addItem(verticalSpacer);

        tabWidget->addTab(DW_Tab_ToDo, QString());
        DW_Tab_Data = new QWidget();
        DW_Tab_Data->setObjectName(QString::fromUtf8("DW_Tab_Data"));
        gridLayout_4 = new QGridLayout(DW_Tab_Data);
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        TE_InfoData = new QTextEdit(DW_Tab_Data);
        TE_InfoData->setObjectName(QString::fromUtf8("TE_InfoData"));

        gridLayout_4->addWidget(TE_InfoData, 0, 0, 1, 1);

        tabWidget->addTab(DW_Tab_Data, QString());
        DW_Tab_Hosts = new QWidget();
        DW_Tab_Hosts->setObjectName(QString::fromUtf8("DW_Tab_Hosts"));
        gridLayout_5 = new QGridLayout(DW_Tab_Hosts);
        gridLayout_5->setContentsMargins(0, 0, 0, 0);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        LW_InfoHosts = new QTextEdit(DW_Tab_Hosts);
        LW_InfoHosts->setObjectName(QString::fromUtf8("LW_InfoHosts"));

        gridLayout_5->addWidget(LW_InfoHosts, 0, 0, 1, 1);

        tabWidget->addTab(DW_Tab_Hosts, QString());

        verticalLayout_5->addWidget(tabWidget);

        DW_Info->setWidget(dockWidgetContents);
        TabViewer->addDockWidget(static_cast<Qt::DockWidgetArea>(1), DW_Info);

        retranslateUi(TabViewer);
        QObject::connect(B_Info, SIGNAL(clicked(bool)), DW_Info, SLOT(setVisible(bool)));

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(TabViewer);
    } // setupUi

    void retranslateUi(QMainWindow *TabViewer)
    {
        TabViewer->setWindowTitle(QApplication::translate("ui_TabViewer", "MainWindow", 0, QApplication::UnicodeUTF8));
        B_Refresh->setText(QString());
        B_Info->setText(QApplication::translate("ui_TabViewer", "i", 0, QApplication::UnicodeUTF8));
        DW_Info->setWindowTitle(QApplication::translate("ui_TabViewer", "Info", 0, QApplication::UnicodeUTF8));
        GB_DW_Tab_ToDo_State->setTitle(QApplication::translate("ui_TabViewer", "State", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("ui_TabViewer", "Window:", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("ui_TabViewer", "From:", 0, QApplication::UnicodeUTF8));
        L_StateWinFrom->setText(QApplication::translate("ui_TabViewer", "N/A", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("ui_TabViewer", "To:", 0, QApplication::UnicodeUTF8));
        L_StateWinTo->setText(QApplication::translate("ui_TabViewer", "N/A", 0, QApplication::UnicodeUTF8));
        L_StateWinStatus->setText(QApplication::translate("ui_TabViewer", "not loaded", 0, QApplication::UnicodeUTF8));
        GB_DW_Tab_ToDo_LoadWin->setTitle(QApplication::translate("ui_TabViewer", "Load window", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("ui_TabViewer", "From", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("ui_TabViewer", "To", 0, QApplication::UnicodeUTF8));
        B_LoadWindow->setText(QApplication::translate("ui_TabViewer", "Load", 0, QApplication::UnicodeUTF8));
        B_dropWindow->setText(QApplication::translate("ui_TabViewer", "Drop", 0, QApplication::UnicodeUTF8));
        GB_DW_Tab_ToDo_Show->setTitle(QApplication::translate("ui_TabViewer", "Show", 0, QApplication::UnicodeUTF8));
        B_ShowMatrix->setText(QApplication::translate("ui_TabViewer", "Matrix", 0, QApplication::UnicodeUTF8));
        GB_showForWin->setTitle(QApplication::translate("ui_TabViewer", "(for window)", 0, QApplication::UnicodeUTF8));
        CB_updFromCurrMtr->setText(QApplication::translate("ui_TabViewer", "update from current matrix", 0, QApplication::UnicodeUTF8));
        B_ShowRow->setText(QApplication::translate("ui_TabViewer", "Row", 0, QApplication::UnicodeUTF8));
        B_ShowColumn->setText(QApplication::translate("ui_TabViewer", "Column", 0, QApplication::UnicodeUTF8));
        B_ShowPair->setText(QApplication::translate("ui_TabViewer", "Pair", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(DW_Tab_ToDo), QApplication::translate("ui_TabViewer", "To Do", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(DW_Tab_Data), QApplication::translate("ui_TabViewer", "Data", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(DW_Tab_Hosts), QApplication::translate("ui_TabViewer", "Hosts", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ui_TabViewer: public Ui_ui_TabViewer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TABVIEWER_H
