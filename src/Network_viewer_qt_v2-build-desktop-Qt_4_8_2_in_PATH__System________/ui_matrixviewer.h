/********************************************************************************
** Form generated from reading UI file 'matrixviewer.ui'
**
** Created: Sun Nov 16 12:42:17 2014
**      by: Qt User Interface Compiler version 4.8.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MATRIXVIEWER_H
#define UI_MATRIXVIEWER_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QRadioButton>
#include <QtGui/QSpinBox>
#include <QtGui/QTextEdit>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>
#include "qwt_plot.h"
#include "qwt_slider.h"

QT_BEGIN_NAMESPACE

class Ui_ui_MatrixViewer
{
public:
    QVBoxLayout *verticalLayout_2;
    QFrame *frame_1;
    QHBoxLayout *horizontalLayout;
    QwtSlider *S_Left;
    QwtPlot *Plot;
    QwtSlider *S_Right;
    QHBoxLayout *horizontalLayout_2;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout;
    QSpinBox *SB_xFrom;
    QSpinBox *SB_yFrom;
    QLineEdit *LE_valFrom;
    QPushButton *B_zoom;
    QSpinBox *SB_xTo;
    QSpinBox *SB_yTo;
    QLineEdit *LE_valTo;
    QLabel *label_2;
    QLabel *label_3;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout;
    QRadioButton *RB_normalizeLocal;
    QRadioButton *RB_normalizeCurrWindow;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QSpinBox *pic_width;
    QLabel *label_4;
    QLabel *label_5;
    QPushButton *pic_save;
    QSpinBox *pic_height;
    QFrame *frame;
    QGridLayout *gridLayout_2;
    QToolButton *tB_showInfo;
    QLabel *label;
    QFrame *line;
    QTextEdit *TB_info;

    void setupUi(QWidget *MatrixViewer)
    {
        if (MatrixViewer->objectName().isEmpty())
            MatrixViewer->setObjectName(QString::fromUtf8("MatrixViewer"));
        MatrixViewer->resize(454, 536);
        MatrixViewer->setMinimumSize(QSize(454, 377));
        verticalLayout_2 = new QVBoxLayout(MatrixViewer);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        frame_1 = new QFrame(MatrixViewer);
        frame_1->setObjectName(QString::fromUtf8("frame_1"));
        horizontalLayout = new QHBoxLayout(frame_1);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        S_Left = new QwtSlider(frame_1);
        S_Left->setObjectName(QString::fromUtf8("S_Left"));
        S_Left->setOrientation(Qt::Vertical);
        S_Left->setScalePosition(QwtSlider::LeftScale);

        horizontalLayout->addWidget(S_Left);

        Plot = new QwtPlot(frame_1);
        Plot->setObjectName(QString::fromUtf8("Plot"));

        horizontalLayout->addWidget(Plot);

        S_Right = new QwtSlider(frame_1);
        S_Right->setObjectName(QString::fromUtf8("S_Right"));
        S_Right->setOrientation(Qt::Vertical);
        S_Right->setScalePosition(QwtSlider::RightScale);

        horizontalLayout->addWidget(S_Right);


        verticalLayout_2->addWidget(frame_1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        groupBox_2 = new QGroupBox(MatrixViewer);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        gridLayout = new QGridLayout(groupBox_2);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        SB_xFrom = new QSpinBox(groupBox_2);
        SB_xFrom->setObjectName(QString::fromUtf8("SB_xFrom"));

        gridLayout->addWidget(SB_xFrom, 0, 1, 1, 1);

        SB_yFrom = new QSpinBox(groupBox_2);
        SB_yFrom->setObjectName(QString::fromUtf8("SB_yFrom"));

        gridLayout->addWidget(SB_yFrom, 0, 2, 1, 1);

        LE_valFrom = new QLineEdit(groupBox_2);
        LE_valFrom->setObjectName(QString::fromUtf8("LE_valFrom"));

        gridLayout->addWidget(LE_valFrom, 0, 3, 1, 1);

        B_zoom = new QPushButton(groupBox_2);
        B_zoom->setObjectName(QString::fromUtf8("B_zoom"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(B_zoom->sizePolicy().hasHeightForWidth());
        B_zoom->setSizePolicy(sizePolicy);

        gridLayout->addWidget(B_zoom, 0, 4, 2, 1);

        SB_xTo = new QSpinBox(groupBox_2);
        SB_xTo->setObjectName(QString::fromUtf8("SB_xTo"));

        gridLayout->addWidget(SB_xTo, 1, 1, 1, 1);

        SB_yTo = new QSpinBox(groupBox_2);
        SB_yTo->setObjectName(QString::fromUtf8("SB_yTo"));

        gridLayout->addWidget(SB_yTo, 1, 2, 1, 1);

        LE_valTo = new QLineEdit(groupBox_2);
        LE_valTo->setObjectName(QString::fromUtf8("LE_valTo"));

        gridLayout->addWidget(LE_valTo, 1, 3, 1, 1);

        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 0, 0, 1, 1);

        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 1, 0, 1, 1);


        horizontalLayout_2->addWidget(groupBox_2);

        groupBox = new QGroupBox(MatrixViewer);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout = new QVBoxLayout(groupBox);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        RB_normalizeLocal = new QRadioButton(groupBox);
        RB_normalizeLocal->setObjectName(QString::fromUtf8("RB_normalizeLocal"));
        RB_normalizeLocal->setChecked(true);

        verticalLayout->addWidget(RB_normalizeLocal);

        RB_normalizeCurrWindow = new QRadioButton(groupBox);
        RB_normalizeCurrWindow->setObjectName(QString::fromUtf8("RB_normalizeCurrWindow"));

        verticalLayout->addWidget(RB_normalizeCurrWindow);


        horizontalLayout_2->addWidget(groupBox);


        verticalLayout_2->addLayout(horizontalLayout_2);

        groupBox_3 = new QGroupBox(MatrixViewer);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        pic_width = new QSpinBox(groupBox_3);
        pic_width->setObjectName(QString::fromUtf8("pic_width"));
        pic_width->setMaximum(10000);

        gridLayout_3->addWidget(pic_width, 0, 1, 1, 1);

        label_4 = new QLabel(groupBox_3);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_3->addWidget(label_4, 0, 0, 1, 1);

        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_3->addWidget(label_5, 1, 0, 1, 1);

        pic_save = new QPushButton(groupBox_3);
        pic_save->setObjectName(QString::fromUtf8("pic_save"));
        sizePolicy.setHeightForWidth(pic_save->sizePolicy().hasHeightForWidth());
        pic_save->setSizePolicy(sizePolicy);

        gridLayout_3->addWidget(pic_save, 0, 2, 2, 1);

        pic_height = new QSpinBox(groupBox_3);
        pic_height->setObjectName(QString::fromUtf8("pic_height"));
        pic_height->setMaximum(10000);
	
        gridLayout_3->addWidget(pic_height, 1, 1, 1, 1);


        verticalLayout_2->addWidget(groupBox_3);

        frame = new QFrame(MatrixViewer);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setMinimumSize(QSize(0, 0));
        frame->setMaximumSize(QSize(16777215, 150));
        frame->setFrameShape(QFrame::NoFrame);
        frame->setFrameShadow(QFrame::Plain);
        frame->setLineWidth(0);
        gridLayout_2 = new QGridLayout(frame);
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        tB_showInfo = new QToolButton(frame);
        tB_showInfo->setObjectName(QString::fromUtf8("tB_showInfo"));
        tB_showInfo->setToolButtonStyle(Qt::ToolButtonFollowStyle);
        tB_showInfo->setAutoRaise(true);
        tB_showInfo->setArrowType(Qt::UpArrow);

        gridLayout_2->addWidget(tB_showInfo, 0, 2, 1, 1);

        label = new QLabel(frame);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_2->addWidget(label, 0, 1, 1, 1);

        line = new QFrame(frame);
        line->setObjectName(QString::fromUtf8("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        gridLayout_2->addWidget(line, 0, 0, 1, 1);

        TB_info = new QTextEdit(frame);
        TB_info->setObjectName(QString::fromUtf8("TB_info"));
        TB_info->setFrameShape(QFrame::StyledPanel);
        TB_info->setFrameShadow(QFrame::Raised);
        TB_info->setLineWidth(0);
        TB_info->setProperty("fixedHeight", QVariant(65));

        gridLayout_2->addWidget(TB_info, 1, 0, 1, 3);

        gridLayout_2->setColumnStretch(0, 1);

        verticalLayout_2->addWidget(frame);


        retranslateUi(MatrixViewer);

        QMetaObject::connectSlotsByName(MatrixViewer);
    } // setupUi

    void retranslateUi(QWidget *MatrixViewer)
    {
        groupBox_2->setTitle(QApplication::translate("MatrixViewer", "Position", 0, QApplication::UnicodeUTF8));
        B_zoom->setText(QApplication::translate("MatrixViewer", "Zoom", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("MatrixViewer", "From (current):", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("MatrixViewer", "To:", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("MatrixViewer", "Normalize", 0, QApplication::UnicodeUTF8));
        RB_normalizeLocal->setText(QApplication::translate("MatrixViewer", "Local", 0, QApplication::UnicodeUTF8));
        RB_normalizeCurrWindow->setText(QApplication::translate("MatrixViewer", "Current window", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("MatrixViewer", "Save Picture", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("MatrixViewer", "Width:", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("MatrixViewer", "Height:", 0, QApplication::UnicodeUTF8));
        pic_save->setText(QApplication::translate("MatrixViewer", "Save", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("MatrixViewer", "info", 0, QApplication::UnicodeUTF8));
        Q_UNUSED(MatrixViewer);
    } // retranslateUi

};

namespace Ui {
    class ui_MatrixViewer: public Ui_ui_MatrixViewer {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MATRIXVIEWER_H
