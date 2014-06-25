#include <QApplication>
#include <QDesktopWidget>
#include "GUI/mainwindow.h"

int main (int argc, char **argv)
{
    QApplication qapp(argc,argv);
    
    QDesktopWidget *const desk=qapp.desktop();
    MainWindow w(desk);
    
    // centre 'w' on the desktop
    const int pos_x=desk->width(),pos_y=desk->height();
    w.move((pos_x<=w.width())? 0 : (pos_x-w.width())>>1u,(pos_y<=w.height())? 0 : (pos_y-w.height())>>1u);
    
    w.Init();
    w.show();
    
    return qapp.exec();
}
