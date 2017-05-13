#ifndef CLUSTVIEWER_H
#define CLUSTVIEWER_H

#include <QWidget>
#include <QString>
#include <vector>
#include <qwt_raster_data.h>
#include <qwt_plot.h>
#include <qwt_plot_spectrogram.h>
#include <qwt_color_map.h>
#include <iostream>
#include <fstream>
#include "core/data_clust.h"

namespace Ui {
class clustviewer;
}

class ClustRaster : public QwtRasterData
{
private:
    std::vector < std::vector <int> > div;
    double nclust;

    Q_DISABLE_COPY(ClustRaster)

public:
    ClustRaster (std::vector < Cluster >, int);
    ClustRaster (std::vector < std::vector <int> > div1, double nclust1) : div(div1), nclust(nclust1) {}

    void PrintDiv(char *filename) {
        std::ofstream a(filename);
        for (int i = 0; i < div.size(); i++) {
            for (int j = 0; j < div[i].size(); j++)
                a << div[i][j] << ' ';
            a << std::endl;
        }
        a.close();
    }

    virtual QwtDoubleInterval range () const { return QwtDoubleInterval(0, nclust); }

    virtual QwtRasterData* copy () const {
        ClustRaster *res = new ClustRaster(div, nclust);
        return res;
    }

    virtual double value (double a, double b) const {
        int x = static_cast<int>(a);
        int y = static_cast<int>(b);
        return div[x][y];
    }
};

class ClustViewer : public QWidget
{
    Q_OBJECT
    
public:
    explicit ClustViewer(QWidget *parent = 0, QString filename = "");

    void DrawClustPlot();

    static ClustViewer* Create(QWidget *parent, QString filename) {
        ClustViewer* ret_win;
        ret_win = new ClustViewer(parent, filename);
        ret_win->reader.readFromFile();
        ret_win->DrawClustPlot();
        return ret_win;
    }
    ~ClustViewer();
    
private:
    Ui::clustviewer *ui;
    ClusterReader reader;

};

#endif // CLUSTVIEWER_H
