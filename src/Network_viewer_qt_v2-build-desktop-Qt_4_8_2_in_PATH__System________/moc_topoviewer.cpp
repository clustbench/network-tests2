/****************************************************************************
** Meta object code from reading C++ file 'topoviewer.h'
**
** Created: Tue Nov 25 11:20:27 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../network_viewer_qt_v2/GUI/topoviewer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'topoviewer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_TVWidget[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      10,    9,    9,    9, 0x08,
      26,    9,    9,    9, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_TVWidget[] = {
    "TVWidget\0\0SaveImageMenu()\0SaveImage()\0"
};

void TVWidget::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        TVWidget *_t = static_cast<TVWidget *>(_o);
        switch (_id) {
        case 0: _t->SaveImageMenu(); break;
        case 1: _t->SaveImage(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData TVWidget::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject TVWidget::staticMetaObject = {
    { &QGLWidget::staticMetaObject, qt_meta_stringdata_TVWidget,
      qt_meta_data_TVWidget, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TVWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TVWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TVWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TVWidget))
        return static_cast<void*>(const_cast< TVWidget*>(this));
    return QGLWidget::qt_metacast(_clname);
}

int TVWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGLWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}
static const uint qt_meta_data_TopologyViewer[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   16,   15,   15, 0x05,
      72,   70,   15,   15, 0x05,
     103,   15,   15,   15, 0x05,

 // slots: signature, parameters, type, tag, flags
     127,   15,   15,   15, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_TopologyViewer[] = {
    "TopologyViewer\0\0,,\0"
    "SendMessToLog(MainWindow::MsgType,QString,QString)\0"
    ",\0TitleChanged(QWidget*,QString)\0"
    "CloseOnEscape(QWidget*)\0Execute()\0"
};

void TopologyViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        TopologyViewer *_t = static_cast<TopologyViewer *>(_o);
        switch (_id) {
        case 0: _t->SendMessToLog((*reinterpret_cast< const MainWindow::MsgType(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3]))); break;
        case 1: _t->TitleChanged((*reinterpret_cast< QWidget*(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2]))); break;
        case 2: _t->CloseOnEscape((*reinterpret_cast< QWidget*(*)>(_a[1]))); break;
        case 3: _t->Execute(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData TopologyViewer::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject TopologyViewer::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_TopologyViewer,
      qt_meta_data_TopologyViewer, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TopologyViewer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TopologyViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TopologyViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TopologyViewer))
        return static_cast<void*>(const_cast< TopologyViewer*>(this));
    return QWidget::qt_metacast(_clname);
}

int TopologyViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 4)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 4;
    }
    return _id;
}

// SIGNAL 0
void TopologyViewer::SendMessToLog(const MainWindow::MsgType _t1, const QString & _t2, const QString & _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void TopologyViewer::TitleChanged(QWidget * _t1, const QString & _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void TopologyViewer::CloseOnEscape(QWidget * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
static const uint qt_meta_data_TopoViewerOpts[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      16,   15,   15,   15, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_TopoViewerOpts[] = {
    "TopoViewerOpts\0\0ShowMaxDistHelp()\0"
};

void TopoViewerOpts::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        TopoViewerOpts *_t = static_cast<TopoViewerOpts *>(_o);
        switch (_id) {
        case 0: _t->ShowMaxDistHelp(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData TopoViewerOpts::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject TopoViewerOpts::staticMetaObject = {
    { &QDialog::staticMetaObject, qt_meta_stringdata_TopoViewerOpts,
      qt_meta_data_TopoViewerOpts, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TopoViewerOpts::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TopoViewerOpts::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TopoViewerOpts::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TopoViewerOpts))
        return static_cast<void*>(const_cast< TopoViewerOpts*>(this));
    return QDialog::qt_metacast(_clname);
}

int TopoViewerOpts::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDialog::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
