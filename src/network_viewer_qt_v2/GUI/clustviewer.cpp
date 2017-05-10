#include "clustviewer.h"
#include "ui_clustviewer.h"

ClustViewer::ClustViewer(QWidget *parent, QString filename) :
    QWidget(parent),
    ui(new Ui::clustviewer),
    reader(filename.toLatin1().data())
{
    ui->setupUi(this);
}

ClustViewer::~ClustViewer()
{
    delete ui;
}
