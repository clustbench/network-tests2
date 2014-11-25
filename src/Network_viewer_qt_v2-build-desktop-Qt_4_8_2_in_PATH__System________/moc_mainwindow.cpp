/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created: Mon Nov 3 13:38:32 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../network_viewer_qt_v2/GUI/mainwindow.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      22,   12,   11,   11, 0x0a,
      73,   71,   11,   11, 0x0a,
     106,   11,   11,   11, 0x0a,
     125,   11,   11,   11, 0x08,
     137,   11,   11,   11, 0x08,
     151,   11,   11,   11, 0x08,
     158,   11,   11,   11, 0x08,
     185,   11,   11,   11, 0x08,
     201,   11,   11,   11, 0x08,
     227,   11,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MainWindow[] = {
    "MainWindow\0\0,sign,msg\0"
    "AddMsgToLog(MainWindow::MsgType,QString,QString)\0"
    ",\0ChangeTabTitle(QWidget*,QString)\0"
    "CloseTab(QWidget*)\0ShowAbout()\0"
    "CloseTab(int)\0Load()\0ToggleWidgetsIn_2D_3D(int)\0"
    "ToggleLog(bool)\0SwitchLanguageToEng(bool)\0"
    "SwitchLanguageToRus(bool)\0"
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MainWindow *_t = static_cast<MainWindow *>(_o);
        switch (_id) {
        case 0: _t->AddMsgToLog((*reinterpret_cast< const MainWindow::MsgType(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3]))); break;
        case 1: _t->ChangeTabTitle((*reinterpret_cast< QWidget*(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2]))); break;
        case 2: _t->CloseTab((*reinterpret_cast< QWidget*(*)>(_a[1]))); break;
        case 3: _t->ShowAbout(); break;
        case 4: _t->CloseTab((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 5: _t->Load(); break;
        case 6: _t->ToggleWidgetsIn_2D_3D((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 7: _t->ToggleLog((*reinterpret_cast< const bool(*)>(_a[1]))); break;
        case 8: _t->SwitchLanguageToEng((*reinterpret_cast< const bool(*)>(_a[1]))); break;
        case 9: _t->SwitchLanguageToRus((*reinterpret_cast< const bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MainWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow,
      qt_meta_data_MainWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MainWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
