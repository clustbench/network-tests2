#include "clustviewer.h"
#include "ui_clustviewer.h"

#include <iostream>

ClustViewer::ClustViewer(QWidget *parent, QString filename) :
    QWidget(parent),
    ui(new Ui::clustviewer),
    reader(filename.toUtf8().constData())
{
    ui->setupUi(this);

}

ClustViewer::~ClustViewer()
{
    delete ui;
}

void ClustViewer::DrawClustPlot()
{
    ui->clustersPlot->setAxisScale(QwtPlot::yLeft,0,reader.getProcNum() - 1, 0);
    ui->clustersPlot->setAxisScale(QwtPlot::xBottom,0,reader.getProcNum() - 1, 0);
    QwtPlotSpectrogram *clSpect = new QwtPlotSpectrogram;
    QwtLinearColorMap colorMap(Qt::darkCyan, Qt::red);
    clSpect->setColorMap(colorMap);

    clSpect->setData(ClustRaster(reader.getClusters(),reader.getProcNum()));
    clSpect->attach(ui->clustersPlot);
    ui->clustersPlot->replot();
}


ClustRaster::ClustRaster(std::vector < Cluster > clusters, int n = 0) :
    QwtRasterData(QwtDoubleRect(0.0, 0.0, n, n)),
    nclust (static_cast<double>(clusters.size()))
{
    this->initRaster(QwtDoubleRect(0.0, 1.0, 0.0, 1.0), QSize(n, n));
    div.resize(n);
    for (int i = 0; i < n; i++)
        div[i].resize(n);
    for (int i = 0; i < clusters.size(); i++) {
        std::vector < std::pair <int, int> > coord = clusters[i].getData();
        int q = coord.size();
        for (int j = 0; j < coord.size(); j++) {
            int p = coord[j].first;
            int a = coord[j].second;
            div[coord[j].first][coord[j].second] = i+1;
        }
    }
    PrintDiv();
}
