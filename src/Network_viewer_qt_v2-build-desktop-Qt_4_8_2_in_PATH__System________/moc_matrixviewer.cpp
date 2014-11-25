/****************************************************************************
** Meta object code from reading C++ file 'matrixviewer.h'
**
** Created: Sun Nov 16 13:15:16 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../network_viewer_qt_v2/GUI/matrixviewer.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'matrixviewer.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MatrixViewer[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       5,       // signalCount

 // signals: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x05,
      33,   13,   13,   13, 0x05,
      51,   13,   13,   13, 0x05,
      81,   77,   13,   13, 0x05,
      94,   77,   13,   13, 0x05,

 // slots: signature, parameters, type, tag, flags
     107,   77,   13,   13, 0x0a,
     131,   13,   13,   13, 0x2a,
     151,   13,   13,   13, 0x0a,
     166,   77,   13,   13, 0x08,
     193,   77,   13,   13, 0x08,
     219,   13,   13,   13, 0x08,
     235,  230,   13,   13, 0x08,
     263,   13,   13,   13, 0x08,
     272,   13,   13,   13, 0x08,
     292,   13,   13,   13, 0x08,
     303,   13,   13,   13, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MatrixViewer[] = {
    "MatrixViewer\0\0GiveInvariant(int)\0"
    "Closing(QWidget*)\0ZoomMatrix(MatrixViewer*)\0"
    "val\0RowChng(int)\0ColChng(int)\0"
    "SetNormalizeToWin(bool)\0SetNormalizeToWin()\0"
    "GetRowAndCol()\0SetRightSldrMinVal(double)\0"
    "SetLeftSldrMaxVal(double)\0ShowInfo()\0"
    "rect\0RectSelected(QwtDoubleRect)\0"
    "SetAim()\0DrawSelectionRect()\0ShowZoom()\0"
    "SaveImage()\0"
};

void MatrixViewer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MatrixViewer *_t = static_cast<MatrixViewer *>(_o);
        switch (_id) {
        case 0: _t->GiveInvariant((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 1: _t->Closing((*reinterpret_cast< QWidget*(*)>(_a[1]))); break;
        case 2: _t->ZoomMatrix((*reinterpret_cast< MatrixViewer*(*)>(_a[1]))); break;
        case 3: _t->RowChng((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->ColChng((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->SetNormalizeToWin((*reinterpret_cast< const bool(*)>(_a[1]))); break;
        case 6: _t->SetNormalizeToWin(); break;
        case 7: _t->GetRowAndCol(); break;
        case 8: _t->SetRightSldrMinVal((*reinterpret_cast< const double(*)>(_a[1]))); break;
        case 9: _t->SetLeftSldrMaxVal((*reinterpret_cast< const double(*)>(_a[1]))); break;
        case 10: _t->ShowInfo(); break;
        case 11: _t->RectSelected((*reinterpret_cast< const QwtDoubleRect(*)>(_a[1]))); break;
        case 12: _t->SetAim(); break;
        case 13: _t->DrawSelectionRect(); break;
        case 14: _t->ShowZoom(); break;
        case 15: _t->SaveImage(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MatrixViewer::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MatrixViewer::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_MatrixViewer,
      qt_meta_data_MatrixViewer, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MatrixViewer::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MatrixViewer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MatrixViewer::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MatrixViewer))
        return static_cast<void*>(const_cast< MatrixViewer*>(this));
    return QWidget::qt_metacast(_clname);
}

int MatrixViewer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    }
    return _id;
}

// SIGNAL 0
void MatrixViewer::GiveInvariant(const int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void MatrixViewer::Closing(QWidget * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void MatrixViewer::ZoomMatrix(MatrixViewer * _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void MatrixViewer::RowChng(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void MatrixViewer::ColChng(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}
QT_END_MOC_NAMESPACE
