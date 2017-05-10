#ifndef CLUSTVIEWER_H
#define CLUSTVIEWER_H

#include <QWidget>
#include "core/data_clust.h"

namespace Ui {
class clustviewer;
}

class ClustViewer : public QWidget
{
    Q_OBJECT
    
public:
    explicit ClustViewer(QWidget *parent = 0, QString filename = "");

    static ClustViewer* Create(QWidget *parent, QString filename) {
        ClustViewer* ret_win;
        ret_win = new ClustViewer(parent, filename);
        ret_win->reader.readFromFile();
        return ret_win;
    }
    ~ClustViewer();
    
private:
    Ui::clustviewer *ui;
    ClusterReader reader;

};

#endif // CLUSTVIEWER_H
