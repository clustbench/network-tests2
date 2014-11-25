/****************************************************************************
** Meta object code from reading C++ file 'tabviewer.h'
**
** Created: Sun Nov 16 13:15:15 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../network_viewer_qt_v2/GUI/tabviewer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'tabviewer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_TabViewer[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      18,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      21,   11,   10,   10, 0x05,

 // slots: signature, parameters, type, tag, flags
      72,   10,   10,   10, 0x08,
      85,   10,   10,   10, 0x08,
      98,   10,   10,   10, 0x08,
     108,   10,   10,   10, 0x08,
     118,   10,   10,   10, 0x08,
     129,   10,   10,   10, 0x08,
     142,   10,   10,   10, 0x08,
     159,  155,   10,   10, 0x08,
     184,   10,   10,   10, 0x08,
     210,  206,   10,   10, 0x08,
     236,  206,   10,   10, 0x08,
     265,  155,   10,   10, 0x0a,
     290,  155,   10,   10, 0x0a,
     307,  155,   10,   10, 0x0a,
     324,   10,   10,   10, 0x0a,
     353,   10,   10,   10, 0x0a,
     382,   10,   10,   10, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_TabViewer[] = {
    "TabViewer\0\0,msg,stat\0"
    "SendMessToLog(MainWindow::MsgType,QString,QString)\0"
    "Initialize()\0ShowMesLen()\0ShowRow()\0"
    "ShowCol()\0ShowPair()\0LoadWindow()\0"
    "DropWindow()\0val\0ChangeMatrNumber(double)\0"
    "ChangeLoadWindowBtn()\0sub\0"
    "DeleteSubWindow(QWidget*)\0"
    "SubActivated(QMdiSubWindow*)\0"
    "SetProgressBarValue(int)\0SetRowValue(int)\0"
    "SetColValue(int)\0NewMatrix_mes(MatrixViewer*)\0"
    "NewMatrix_row(MatrixViewer*)\0"
    "NewMatrix_col(MatrixViewer*)\0"
};

void TabViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        TabViewer *_t = static_cast<TabViewer *>(_o);
        switch (_id) {
        case 0: _t->SendMessToLog((*reinterpret_cast< const MainWindow::MsgType(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3]))); break;
        case 1: _t->Initialize(); break;
        case 2: _t->ShowMesLen(); break;
        case 3: _t->ShowRow(); break;
        case 4: _t->ShowCol(); break;
        case 5: _t->ShowPair(); break;
        case 6: _t->LoadWindow(); break;
        case 7: _t->DropWindow(); break;
        case 8: _t->ChangeMatrNumber((*reinterpret_cast< const double(*)>(_a[1]))); break;
        case 9: _t->ChangeLoadWindowBtn(); break;
        case 10: _t->DeleteSubWindow((*reinterpret_cast< QWidget*(*)>(_a[1]))); break;
        case 11: _t->SubActivated((*reinterpret_cast< QMdiSubWindow*(*)>(_a[1]))); break;
        case 12: _t->SetProgressBarValue((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 13: _t->SetRowValue((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 14: _t->SetColValue((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 15: _t->NewMatrix_mes((*reinterpret_cast< MatrixViewer*(*)>(_a[1]))); break;
        case 16: _t->NewMatrix_row((*reinterpret_cast< MatrixViewer*(*)>(_a[1]))); break;
        case 17: _t->NewMatrix_col((*reinterpret_cast< MatrixViewer*(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData TabViewer::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject TabViewer::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_TabViewer,
      qt_meta_data_TabViewer, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TabViewer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TabViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TabViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TabViewer))
        return static_cast<void*>(const_cast< TabViewer*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int TabViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 18)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 18;
    }
    return _id;
}

// SIGNAL 0
void TabViewer::SendMessToLog(const MainWindow::MsgType _t1, const QString & _t2, const QString & _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
