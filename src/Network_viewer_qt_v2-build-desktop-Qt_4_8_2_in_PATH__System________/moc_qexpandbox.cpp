/****************************************************************************
** Meta object code from reading C++ file 'qexpandbox.h'
**
** Created: Mon Nov 3 13:38:28 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../network_viewer_qt_v2/GUI/qexpandbox.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'qexpandbox.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QExpandBox[] = {

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
      16,   12,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_QExpandBox[] = {
    "QExpandBox\0\0ind\0operate(uint)\0"
};

void QExpandBox::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        QExpandBox *_t = static_cast<QExpandBox *>(_o);
        switch (_id) {
        case 0: _t->operate((*reinterpret_cast< const uint(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData QExpandBox::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject QExpandBox::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_QExpandBox,
      qt_meta_data_QExpandBox, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QExpandBox::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QExpandBox::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QExpandBox::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QExpandBox))
        return static_cast<void*>(const_cast< QExpandBox*>(this));
    return QWidget::qt_metacast(_clname);
}

int QExpandBox::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_MasterButton[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x04,

 // slots: signature, parameters, type, tag, flags
      32,   13,   13,   13, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_MasterButton[] = {
    "MasterButton\0\0iWasClicked(uint)\0"
    "operate()\0"
};

void MasterButton::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MasterButton *_t = static_cast<MasterButton *>(_o);
        switch (_id) {
        case 0: _t->iWasClicked((*reinterpret_cast< const uint(*)>(_a[1]))); break;
        case 1: _t->operate(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData MasterButton::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject MasterButton::staticMetaObject = {
    { &QToolButton::staticMetaObject, qt_meta_stringdata_MasterButton,
      qt_meta_data_MasterButton, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MasterButton::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MasterButton::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MasterButton::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MasterButton))
        return static_cast<void*>(const_cast< MasterButton*>(this));
    return QToolButton::qt_metacast(_clname);
}

int MasterButton::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QToolButton::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}

// SIGNAL 0
void MasterButton::iWasClicked(const unsigned int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
