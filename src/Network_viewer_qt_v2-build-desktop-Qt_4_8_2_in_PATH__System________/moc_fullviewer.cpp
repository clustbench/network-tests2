/****************************************************************************
** Meta object code from reading C++ file 'fullviewer.h'
**
** Created: Tue Nov 25 11:20:25 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../network_viewer_qt_v2/GUI/fullviewer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'fullviewer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_FullViewer[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       3,       // signalCount

 // signals: signature, parameters, type, tag, flags
      15,   12,   11,   11, 0x05,
      68,   66,   11,   11, 0x05,
      99,   11,   11,   11, 0x05,

 // slots: signature, parameters, type, tag, flags
     137,  123,   11,   11, 0x0a,
     162,   11,   11,   11, 0x08,
     174,   11,   11,   11, 0x08,
     189,   11,   11,   11, 0x08,
     205,   11,   11,   11, 0x08,
     217,   11,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_FullViewer[] = {
    "FullViewer\0\0,,\0"
    "SendMessToLog(MainWindow::MsgType,QString,QString)\0"
    ",\0TitleChanged(QWidget*,QString)\0"
    "CloseOnEscape(QWidget*)\0cosa,positive\0"
    "OXY_Rotation(float,bool)\0RenderBox()\0"
    "ShowControls()\0SaveImageMenu()\0"
    "SaveImage()\0ShowHostsInfo()\0"
};

void FullViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        FullViewer *_t = static_cast<FullViewer *>(_o);
        switch (_id) {
        case 0: _t->SendMessToLog((*reinterpret_cast< const MainWindow::MsgType(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2])),(*reinterpret_cast< const QString(*)>(_a[3]))); break;
        case 1: _t->TitleChanged((*reinterpret_cast< QWidget*(*)>(_a[1])),(*reinterpret_cast< const QString(*)>(_a[2]))); break;
        case 2: _t->CloseOnEscape((*reinterpret_cast< QWidget*(*)>(_a[1]))); break;
        case 3: _t->OXY_Rotation((*reinterpret_cast< const float(*)>(_a[1])),(*reinterpret_cast< const bool(*)>(_a[2]))); break;
        case 4: _t->RenderBox(); break;
        case 5: _t->ShowControls(); break;
        case 6: _t->SaveImageMenu(); break;
        case 7: _t->SaveImage(); break;
        case 8: _t->ShowHostsInfo(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData FullViewer::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject FullViewer::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_FullViewer,
      qt_meta_data_FullViewer, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &FullViewer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *FullViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *FullViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_FullViewer))
        return static_cast<void*>(const_cast< FullViewer*>(this));
    return QWidget::qt_metacast(_clname);
}

int FullViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 9)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void FullViewer::SendMessToLog(const MainWindow::MsgType _t1, const QString & _t2, const QString & _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void FullViewer::TitleChanged(QWidget * _t1, const QString & _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void FullViewer::CloseOnEscape(QWidget * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}
static const uint qt_meta_data_RotOXYButton[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      16,   14,   13,   13, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_RotOXYButton[] = {
    "RotOXYButton\0\0,\0NeedRender(float,bool)\0"
};

void RotOXYButton::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        RotOXYButton *_t = static_cast<RotOXYButton *>(_o);
        switch (_id) {
        case 0: _t->NeedRender((*reinterpret_cast< float(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData RotOXYButton::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject RotOXYButton::staticMetaObject = {
    { &QPushButton::staticMetaObject, qt_meta_stringdata_RotOXYButton,
      qt_meta_data_RotOXYButton, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &RotOXYButton::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *RotOXYButton::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *RotOXYButton::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_RotOXYButton))
        return static_cast<void*>(const_cast< RotOXYButton*>(this));
    return QPushButton::qt_metacast(_clname);
}

int RotOXYButton::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QPushButton::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void RotOXYButton::NeedRender(float _t1, bool _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
static const uint qt_meta_data_HostsBrowser[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_HostsBrowser[] = {
    "HostsBrowser\0"
};

void HostsBrowser::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObjectExtraData HostsBrowser::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject HostsBrowser::staticMetaObject = {
    { &QTextEdit::staticMetaObject, qt_meta_stringdata_HostsBrowser,
      qt_meta_data_HostsBrowser, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &HostsBrowser::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *HostsBrowser::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *HostsBrowser::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_HostsBrowser))
        return static_cast<void*>(const_cast< HostsBrowser*>(this));
    return QTextEdit::qt_metacast(_clname);
}

int HostsBrowser::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QTextEdit::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE
