/********************************************************************************
** Form generated from reading UI file 'topoviewer.ui'
**
** Created: Mon Nov 3 13:37:53 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TOPOVIEWER_H
#define UI_TOPOVIEWER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDialog>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFrame>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpinBox>
#include <QtGui/QTabWidget>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TopoOptions
{
public:
    QHBoxLayout *horizontalLayout;
    QTabWidget *mainWidget;
    QWidget *tabRetr;
    QLabel *shmLbl;
    QDoubleSpinBox *shmEpsSB;
    QLabel *dupLbl;
    QDoubleSpinBox *dupEpsSB;
    QPushButton *OK1;
    QWidget *tabBld;
    QCheckBox *maxDistCB;
    QFrame *frame_2;
    QLabel *impValLbl;
    QFrame *frame;
    QRadioButton *impValManRB;
    QRadioButton *impValAutoRB;
    QDoubleSpinBox *impactValSB;
    QFrame *frame_3;
    QLabel *impValAutoLbl;
    QSpinBox *impValAutoNoTSB;
    QPushButton *OK2;
    QPushButton *maxDistHelpPB;
    QLabel *nonExEdgLbl;
    QSpinBox *nonExEdgSB;
    QFrame *line;
    QFrame *frame_4;
    QSpinBox *mesLenSB;
    QRadioButton *srcMesLenRB;
    QRadioButton *srcAvgRB;
    QRadioButton *srcMedRB;
    QLabel *valsEdgLenLbl;
    QWidget *tabDrw;
    QCheckBox *hideEdgesCB;
    QSpinBox *probabSB;
    QCheckBox *immRedrCB;
    QCheckBox *showVertLblsCB;
    QPushButton *OK3;
    QFrame *line_3;

    void setupUi(QDialog *TopoOptions)
    {
        if (TopoOptions->objectName().isEmpty())
            TopoOptions->setObjectName(QString::fromUtf8("TopoOptions"));
        TopoOptions->resize(464, 421);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(TopoOptions->sizePolicy().hasHeightForWidth());
        TopoOptions->setSizePolicy(sizePolicy);
        QIcon icon;
        icon.addFile(QString::fromUtf8("../img/logotype.png"), QSize(), QIcon::Normal, QIcon::Off);
        TopoOptions->setWindowIcon(icon);
        horizontalLayout = new QHBoxLayout(TopoOptions);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        mainWidget = new QTabWidget(TopoOptions);
        mainWidget->setObjectName(QString::fromUtf8("mainWidget"));
        mainWidget->setUsesScrollButtons(false);
        tabRetr = new QWidget();
        tabRetr->setObjectName(QString::fromUtf8("tabRetr"));
        shmLbl = new QLabel(tabRetr);
        shmLbl->setObjectName(QString::fromUtf8("shmLbl"));
        shmLbl->setGeometry(QRect(30, 30, 281, 41));
        shmLbl->setAlignment(Qt::AlignJustify|Qt::AlignVCenter);
        shmLbl->setWordWrap(true);
        shmEpsSB = new QDoubleSpinBox(tabRetr);
        shmEpsSB->setObjectName(QString::fromUtf8("shmEpsSB"));
        shmEpsSB->setGeometry(QRect(340, 40, 62, 31));
        shmEpsSB->setAlignment(Qt::AlignCenter);
        shmEpsSB->setMinimum(0);
        shmEpsSB->setMaximum(0.5);
        shmEpsSB->setSingleStep(0.01);
        shmEpsSB->setValue(0.01);
        dupLbl = new QLabel(tabRetr);
        dupLbl->setObjectName(QString::fromUtf8("dupLbl"));
        dupLbl->setGeometry(QRect(30, 130, 281, 61));
        dupLbl->setAlignment(Qt::AlignJustify|Qt::AlignVCenter);
        dupLbl->setWordWrap(true);
        dupEpsSB = new QDoubleSpinBox(tabRetr);
        dupEpsSB->setObjectName(QString::fromUtf8("dupEpsSB"));
        dupEpsSB->setGeometry(QRect(340, 150, 62, 31));
        dupEpsSB->setMaximum(1);
        dupEpsSB->setSingleStep(0.01);
        dupEpsSB->setValue(0.01);
        OK1 = new QPushButton(tabRetr);
        OK1->setObjectName(QString::fromUtf8("OK1"));
        OK1->setGeometry(QRect(355, 335, 81, 31));
        OK1->setText(QString::fromUtf8("OK"));
        OK1->setAutoDefault(false);
        OK1->setDefault(true);
        mainWidget->addTab(tabRetr, QString());
        tabBld = new QWidget();
        tabBld->setObjectName(QString::fromUtf8("tabBld"));
        maxDistCB = new QCheckBox(tabBld);
        maxDistCB->setObjectName(QString::fromUtf8("maxDistCB"));
        maxDistCB->setGeometry(QRect(10, 201, 301, 31));
        maxDistCB->setChecked(true);
        frame_2 = new QFrame(tabBld);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setGeometry(QRect(30, 240, 401, 121));
        frame_2->setFrameShape(QFrame::NoFrame);
        frame_2->setFrameShadow(QFrame::Plain);
        impValLbl = new QLabel(frame_2);
        impValLbl->setObjectName(QString::fromUtf8("impValLbl"));
        impValLbl->setGeometry(QRect(0, 0, 391, 17));
        frame = new QFrame(frame_2);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setGeometry(QRect(30, 20, 371, 65));
        frame->setFrameShape(QFrame::NoFrame);
        frame->setFrameShadow(QFrame::Plain);
        impValManRB = new QRadioButton(frame);
        impValManRB->setObjectName(QString::fromUtf8("impValManRB"));
        impValManRB->setGeometry(QRect(0, 10, 141, 22));
        impValManRB->setChecked(true);
        impValAutoRB = new QRadioButton(frame);
        impValAutoRB->setObjectName(QString::fromUtf8("impValAutoRB"));
        impValAutoRB->setEnabled(false);
        impValAutoRB->setGeometry(QRect(0, 40, 371, 22));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(impValAutoRB->sizePolicy().hasHeightForWidth());
        impValAutoRB->setSizePolicy(sizePolicy1);
        impValAutoRB->setChecked(false);
        impactValSB = new QDoubleSpinBox(frame);
        impactValSB->setObjectName(QString::fromUtf8("impactValSB"));
        impactValSB->setGeometry(QRect(160, 8, 70, 30));
        impactValSB->setDecimals(3);
        impactValSB->setMaximum(1);
        impactValSB->setSingleStep(0.001);
        impactValSB->setValue(1);
        frame_3 = new QFrame(frame_2);
        frame_3->setObjectName(QString::fromUtf8("frame_3"));
        frame_3->setEnabled(false);
        frame_3->setGeometry(QRect(30, 85, 371, 30));
        frame_3->setFrameShape(QFrame::NoFrame);
        frame_3->setFrameShadow(QFrame::Plain);
        impValAutoLbl = new QLabel(frame_3);
        impValAutoLbl->setObjectName(QString::fromUtf8("impValAutoLbl"));
        impValAutoLbl->setGeometry(QRect(40, 5, 121, 20));
        impValAutoNoTSB = new QSpinBox(frame_3);
        impValAutoNoTSB->setObjectName(QString::fromUtf8("impValAutoNoTSB"));
        impValAutoNoTSB->setGeometry(QRect(170, 0, 60, 30));
        impValAutoNoTSB->setMinimum(1);
        impValAutoNoTSB->setMaximum(10);
        OK2 = new QPushButton(tabBld);
        OK2->setObjectName(QString::fromUtf8("OK2"));
        OK2->setGeometry(QRect(355, 335, 81, 31));
        OK2->setText(QString::fromUtf8("OK"));
        OK2->setAutoDefault(false);
        OK2->setDefault(true);
        maxDistHelpPB = new QPushButton(tabBld);
        maxDistHelpPB->setObjectName(QString::fromUtf8("maxDistHelpPB"));
        maxDistHelpPB->setGeometry(QRect(310, 205, 25, 25));
        maxDistHelpPB->setText(QString::fromUtf8("?"));
        nonExEdgLbl = new QLabel(tabBld);
        nonExEdgLbl->setObjectName(QString::fromUtf8("nonExEdgLbl"));
        nonExEdgLbl->setGeometry(QRect(20, 140, 321, 41));
        nonExEdgLbl->setWordWrap(true);
        nonExEdgSB = new QSpinBox(tabBld);
        nonExEdgSB->setObjectName(QString::fromUtf8("nonExEdgSB"));
        nonExEdgSB->setGeometry(QRect(350, 145, 71, 31));
        nonExEdgSB->setSuffix(QString::fromUtf8("%"));
        nonExEdgSB->setMinimum(1);
        nonExEdgSB->setMaximum(100);
        line = new QFrame(tabBld);
        line->setObjectName(QString::fromUtf8("line"));
        line->setGeometry(QRect(10, 180, 421, 16));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);
        frame_4 = new QFrame(tabBld);
        frame_4->setObjectName(QString::fromUtf8("frame_4"));
        frame_4->setGeometry(QRect(10, 20, 421, 111));
        frame_4->setFrameShape(QFrame::Box);
        frame_4->setFrameShadow(QFrame::Sunken);
        mesLenSB = new QSpinBox(frame_4);
        mesLenSB->setObjectName(QString::fromUtf8("mesLenSB"));
        mesLenSB->setGeometry(QRect(305, 75, 110, 30));
        QFont font;
        font.setBold(false);
        font.setWeight(50);
        mesLenSB->setFont(font);
        mesLenSB->setMinimum(0);
        mesLenSB->setMaximum(0);
        srcMesLenRB = new QRadioButton(frame_4);
        srcMesLenRB->setObjectName(QString::fromUtf8("srcMesLenRB"));
        srcMesLenRB->setGeometry(QRect(10, 78, 291, 21));
        srcMesLenRB->setFont(font);
        srcMesLenRB->setChecked(true);
        srcAvgRB = new QRadioButton(frame_4);
        srcAvgRB->setObjectName(QString::fromUtf8("srcAvgRB"));
        srcAvgRB->setGeometry(QRect(10, 18, 331, 21));
        srcAvgRB->setFont(font);
        srcAvgRB->setChecked(false);
        srcMedRB = new QRadioButton(frame_4);
        srcMedRB->setObjectName(QString::fromUtf8("srcMedRB"));
        srcMedRB->setGeometry(QRect(10, 48, 221, 21));
        srcMedRB->setFont(font);
        valsEdgLenLbl = new QLabel(tabBld);
        valsEdgLenLbl->setObjectName(QString::fromUtf8("valsEdgLenLbl"));
        valsEdgLenLbl->setGeometry(QRect(30, 11, 191, 17));
        sizePolicy1.setHeightForWidth(valsEdgLenLbl->sizePolicy().hasHeightForWidth());
        valsEdgLenLbl->setSizePolicy(sizePolicy1);
        valsEdgLenLbl->setAutoFillBackground(true);
        valsEdgLenLbl->setTextFormat(Qt::PlainText);
        mainWidget->addTab(tabBld, QString());
        tabDrw = new QWidget();
        tabDrw->setObjectName(QString::fromUtf8("tabDrw"));
        hideEdgesCB = new QCheckBox(tabDrw);
        hideEdgesCB->setObjectName(QString::fromUtf8("hideEdgesCB"));
        hideEdgesCB->setGeometry(QRect(10, 20, 231, 51));
        hideEdgesCB->setFont(font);
        probabSB = new QSpinBox(tabDrw);
        probabSB->setObjectName(QString::fromUtf8("probabSB"));
        probabSB->setEnabled(false);
        probabSB->setGeometry(QRect(240, 35, 71, 25));
        probabSB->setFont(font);
        probabSB->setSuffix(QString::fromUtf8(" %"));
        probabSB->setMinimum(1);
        probabSB->setMaximum(100);
        immRedrCB = new QCheckBox(tabDrw);
        immRedrCB->setObjectName(QString::fromUtf8("immRedrCB"));
        immRedrCB->setGeometry(QRect(320, 27, 121, 41));
        showVertLblsCB = new QCheckBox(tabDrw);
        showVertLblsCB->setObjectName(QString::fromUtf8("showVertLblsCB"));
        showVertLblsCB->setGeometry(QRect(10, 100, 361, 22));
        sizePolicy1.setHeightForWidth(showVertLblsCB->sizePolicy().hasHeightForWidth());
        showVertLblsCB->setSizePolicy(sizePolicy1);
        showVertLblsCB->setChecked(true);
        OK3 = new QPushButton(tabDrw);
        OK3->setObjectName(QString::fromUtf8("OK3"));
        OK3->setGeometry(QRect(355, 335, 81, 31));
        OK3->setText(QString::fromUtf8("OK"));
        OK3->setAutoDefault(false);
        OK3->setDefault(true);
        line_3 = new QFrame(tabDrw);
        line_3->setObjectName(QString::fromUtf8("line_3"));
        line_3->setGeometry(QRect(10, 80, 421, 16));
        line_3->setFrameShape(QFrame::HLine);
        line_3->setFrameShadow(QFrame::Sunken);
        mainWidget->addTab(tabDrw, QString());

        horizontalLayout->addWidget(mainWidget);


        retranslateUi(TopoOptions);
        QObject::connect(maxDistCB, SIGNAL(toggled(bool)), frame_2, SLOT(setEnabled(bool)));
        QObject::connect(impValAutoRB, SIGNAL(toggled(bool)), frame_3, SLOT(setEnabled(bool)));
        QObject::connect(OK1, SIGNAL(clicked()), TopoOptions, SLOT(accept()));
        QObject::connect(OK2, SIGNAL(clicked()), TopoOptions, SLOT(accept()));
        QObject::connect(OK3, SIGNAL(clicked()), TopoOptions, SLOT(accept()));
        QObject::connect(hideEdgesCB, SIGNAL(toggled(bool)), probabSB, SLOT(setEnabled(bool)));

        mainWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(TopoOptions);
    } // setupUi

    void retranslateUi(QDialog *TopoOptions)
    {
        TopoOptions->setWindowTitle(QApplication::translate("TopoOptions", "Topology viewer options", 0, QApplication::UnicodeUTF8));
        shmLbl->setText(QApplication::translate("TopoOptions", "Dispersion of distances between processors in shared memory:", 0, QApplication::UnicodeUTF8));
        dupLbl->setText(QApplication::translate("TopoOptions", "Merge two simpex channels into one duplex channel if a dispersion of values is not greater than:", 0, QApplication::UnicodeUTF8));
        mainWidget->setTabText(mainWidget->indexOf(tabRetr), QApplication::translate("TopoOptions", "Retrieving", 0, QApplication::UnicodeUTF8));
        maxDistCB->setText(QApplication::translate("TopoOptions", "Maximize distances between\n"
"unconnected vertices", 0, QApplication::UnicodeUTF8));
        impValLbl->setText(QApplication::translate("TopoOptions", "- minimize impact on the other parts of the graph:", 0, QApplication::UnicodeUTF8));
        impValManRB->setText(QApplication::translate("TopoOptions", "set manually:", 0, QApplication::UnicodeUTF8));
        impValAutoRB->setText(QApplication::translate("TopoOptions", "auto detect (may be slow!)", 0, QApplication::UnicodeUTF8));
        impValAutoLbl->setText(QApplication::translate("TopoOptions", "number of tries:", 0, QApplication::UnicodeUTF8));
        nonExEdgLbl->setText(QApplication::translate("TopoOptions", "Consider non-existent edges with existence probability less than", 0, QApplication::UnicodeUTF8));
        srcMesLenRB->setText(QApplication::translate("TopoOptions", "defined by message length:", 0, QApplication::UnicodeUTF8));
        srcAvgRB->setText(QApplication::translate("TopoOptions", "simple average of all values", 0, QApplication::UnicodeUTF8));
        srcMedRB->setText(QApplication::translate("TopoOptions", "median of all values", 0, QApplication::UnicodeUTF8));
        valsEdgLenLbl->setText(QApplication::translate("TopoOptions", " Values for edges' lengths:", 0, QApplication::UnicodeUTF8));
        mainWidget->setTabText(mainWidget->indexOf(tabBld), QApplication::translate("TopoOptions", "Building", 0, QApplication::UnicodeUTF8));
        hideEdgesCB->setText(QApplication::translate("TopoOptions", "Hide edges with existence\n"
"probability less than", 0, QApplication::UnicodeUTF8));
        immRedrCB->setText(QApplication::translate("TopoOptions", "immediate\n"
"redraw", 0, QApplication::UnicodeUTF8));
        showVertLblsCB->setText(QApplication::translate("TopoOptions", "Show host names near the vertices", 0, QApplication::UnicodeUTF8));
        mainWidget->setTabText(mainWidget->indexOf(tabDrw), QApplication::translate("TopoOptions", "Drawing", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class TopoOptions: public Ui_TopoOptions {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TOPOVIEWER_H
