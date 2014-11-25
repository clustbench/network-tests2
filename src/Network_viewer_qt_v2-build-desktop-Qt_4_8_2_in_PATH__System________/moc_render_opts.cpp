/****************************************************************************
** Meta object code from reading C++ file 'render_opts.h'
**
** Created: Mon Nov 3 13:38:29 2014
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../network_viewer_qt_v2/GUI/render_opts.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'render_opts.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_RenderOpts[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x08,
      27,   11,   11,   11, 0x08,
      49,   11,   11,   11, 0x08,
      70,   11,   11,   11, 0x08,
      91,   11,   11,   11, 0x08,
     117,   11,   11,   11, 0x08,
     141,   11,   11,   11, 0x08,
     169,   11,   11,   11, 0x08,
     197,   11,   11,   11, 0x08,
     219,   11,   11,   11, 0x08,
     245,   11,   11,   11, 0x08,
     272,   11,   11,   11, 0x08,
     299,   11,   11,   11, 0x08,
     326,   11,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_RenderOpts[] = {
    "RenderOpts\0\0ChangePtRepr()\0"
    "ShowReprSpheresInfo()\0ShowReprLightsInfo()\0"
    "SetDepthConstraint()\0ShowDepthConstraintInfo()\0"
    "ShowClrStretchingInfo()\0"
    "AdjustClrStretchingMin(int)\0"
    "AdjustClrStretchingMax(int)\0"
    "ShowVolBuildingInfo()\0EnterVolBuildingMode(int)\0"
    "AdjustVolBuildingMin1(int)\0"
    "AdjustVolBuildingMax1(int)\0"
    "AdjustVolBuildingMin2(int)\0"
    "AdjustVolBuildingMax2(int)\0"
};

void RenderOpts::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        RenderOpts *_t = static_cast<RenderOpts *>(_o);
        switch (_id) {
        case 0: _t->ChangePtRepr(); break;
        case 1: _t->ShowReprSpheresInfo(); break;
        case 2: _t->ShowReprLightsInfo(); break;
        case 3: _t->SetDepthConstraint(); break;
        case 4: _t->ShowDepthConstraintInfo(); break;
        case 5: _t->ShowClrStretchingInfo(); break;
        case 6: _t->AdjustClrStretchingMin((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 7: _t->AdjustClrStretchingMax((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 8: _t->ShowVolBuildingInfo(); break;
        case 9: _t->EnterVolBuildingMode((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 10: _t->AdjustVolBuildingMin1((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 11: _t->AdjustVolBuildingMax1((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 12: _t->AdjustVolBuildingMin2((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 13: _t->AdjustVolBuildingMax2((*reinterpret_cast< const int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData RenderOpts::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject RenderOpts::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_RenderOpts,
      qt_meta_data_RenderOpts, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &RenderOpts::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *RenderOpts::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *RenderOpts::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_RenderOpts))
        return static_cast<void*>(const_cast< RenderOpts*>(this));
    return QWidget::qt_metacast(_clname);
}

int RenderOpts::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
