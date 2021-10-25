#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/experimental_dtype_api.h>
#include "structmember.h"
#include <math.h>

typedef struct
{
    PyObject_HEAD double value;
} CustomFloat;

static PyTypeObject CustomFloat_Type;

static int
CustomFloat_Check(PyObject *object)
{
    return PyObject_IsInstance(object, (PyObject *)&CustomFloat_Type);
}

static PyObject *
CustomFloat_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    CustomFloat *self;
    self = (CustomFloat *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

static int
CustomFloat_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    Py_ssize_t size = PyTuple_Size(args);
    CustomFloat *c = (CustomFloat *)self;
    if (kwds && PyDict_Size(kwds))
    {
        PyErr_SetString(PyExc_TypeError, "custom takes no keywords");
        return -1;
    }
    c->value = 0.0;
    if (size == 0)
    {
        return 0;
    }
    else if (size == 1)
    {
        if (PyArg_ParseTuple(args, "d", &c->value))
        {
            return 0;
        }
    }
    PyErr_SetString(PyExc_TypeError, "custom takes one or two arguments");
    return -1;
}

static PyObject *
CustomFloat_repr(PyObject *self)
{
    char str[128];
    CustomFloat *c = (CustomFloat *)self;
    sprintf(str, "%.15g", c->value);
    return PyUnicode_FromString(str);
}

static PyObject *
CustomFloat_str(PyObject *self)
{
    char str[128];
    CustomFloat *c = (CustomFloat *)self;
    sprintf(str, "%.15g", c->value);
    return PyUnicode_FromString(str);
}

static CustomFloat *
add(CustomFloat *a, CustomFloat *b)
{
    double sum;
    CustomFloat *ret = (CustomFloat *)CustomFloat_Type.tp_alloc(&CustomFloat_Type, 0);
    sum = a->value + b->value;
    ret->value = sum;
    return ret;
}

static PyObject *
CustomFloat_add(PyObject *a, PyObject *b)
{
    CustomFloat *c;
    CustomFloat *d;
    double sum;
    CustomFloat *ret = (CustomFloat *)CustomFloat_Type.tp_alloc(&CustomFloat_Type, 0);
    if (CustomFloat_Check(a))
    {
        c = (CustomFloat *)a;
        if (PyFloat_Check(b))
        {
            sum = c->value + PyFloat_AsDouble(b);
            ret->value = sum;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(b))
        {
            sum = c->value + PyLong_AsLong(b);
            ret->value = sum;
            return (PyObject *)ret;
        }
        else if (CustomFloat_Check(b))
        {
            d = (CustomFloat *)b;
            sum = c->value + d->value;
            ret->value = sum;
            return (PyObject *)ret;
        }
    }
    else
    {
        c = (CustomFloat *)b;
        if (PyFloat_Check(a))
        {
            sum = PyFloat_AsDouble(a) + c->value;
            ret->value = sum;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(a))
        {
            sum = PyLong_AsLong(a) + c->value;
            ret->value = sum;
            return (PyObject *)ret;
        }
    }
    return Py_NotImplemented;
}

static PyObject *
CustomFloat_subtract(PyObject *a, PyObject *b)
{
    CustomFloat *c;
    CustomFloat *d;
    double sub;
    CustomFloat *ret = (CustomFloat *)CustomFloat_Type.tp_alloc(&CustomFloat_Type, 0);
    if (CustomFloat_Check(a))
    {
        c = (CustomFloat *)a;
        if (PyFloat_Check(b))
        {
            sub = c->value - PyFloat_AsDouble(b);
            ret->value = sub;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(b))
        {
            sub = c->value - PyLong_AsLong(b);
            ret->value = sub;
            return (PyObject *)ret;
        }
        else if (CustomFloat_Check(b))
        {
            d = (CustomFloat *)b;
            sub = c->value - d->value;
            ret->value = sub;
            return (PyObject *)ret;
        }
    }
    else
    {
        c = (CustomFloat *)b;
        if (PyFloat_Check(a))
        {
            sub = PyFloat_AsDouble(a) - c->value;
            ret->value = sub;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(a))
        {
            sub = PyLong_AsLong(a) - c->value;
            ret->value = sub;
            return (PyObject *)ret;
        }
    }
    return Py_NotImplemented;
}

static PyObject *
CustomFloat_multiply(PyObject *a, PyObject *b)
{
    CustomFloat *c;
    CustomFloat *d;
    double mul;
    CustomFloat *ret = (CustomFloat *)CustomFloat_Type.tp_alloc(&CustomFloat_Type, 0);
    if (CustomFloat_Check(a))
    {
        c = (CustomFloat *)a;
        if (PyFloat_Check(b))
        {
            mul = c->value * PyFloat_AsDouble(b);
            ret->value = mul;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(b))
        {
            mul = c->value * PyLong_AsLong(b);
            ret->value = mul;
            return (PyObject *)ret;
        }
        else if (CustomFloat_Check(b))
        {
            d = (CustomFloat *)b;
            mul = c->value * d->value;
            ret->value = mul;
            return (PyObject *)ret;
        }
    }
    else
    {
        c = (CustomFloat *)b;
        if (PyFloat_Check(a))
        {
            mul = PyFloat_AsDouble(a) * c->value;
            ret->value = mul;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(a))
        {
            mul = PyLong_AsLong(a) * c->value;
            ret->value = mul;
            return (PyObject *)ret;
        }
    }
    return Py_NotImplemented;
}

static PyObject *
CustomFloat_divide(PyObject *a, PyObject *b)
{
    CustomFloat *c;
    CustomFloat *d;
    double div;
    CustomFloat *ret = (CustomFloat *)CustomFloat_Type.tp_alloc(&CustomFloat_Type, 0);
    if (CustomFloat_Check(a))
    {
        c = (CustomFloat *)a;
        if (PyFloat_Check(b))
        {
            div = c->value / PyFloat_AsDouble(b);
            ret->value = div;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(b))
        {
            div = c->value / PyLong_AsLong(b);
            ret->value = div;
            return (PyObject *)ret;
        }
        else if (CustomFloat_Check(b))
        {
            d = (CustomFloat *)b;
            div = c->value / d->value;
            ret->value = div;
            return (PyObject *)ret;
        }
    }
    else
    {
        c = (CustomFloat *)b;
        if (PyFloat_Check(a))
        {
            div = PyFloat_AsDouble(a) / c->value;
            ret->value = div;
            return (PyObject *)ret;
        }
        else if (PyLong_Check(a))
        {
            div = PyLong_AsLong(a) / c->value;
            ret->value = div;
            return (PyObject *)ret;
        }
    }
    return Py_NotImplemented;
}

static PyObject *
CustomFloat_nonzero(PyObject *a)
{
    CustomFloat *c = (CustomFloat *)a;
    int ret;
    if (c->value == 0)
    {
        ret = 0;
    }
    else
    {
        ret = 1;
    }
    return PyBool_FromLong(ret);
}

PyMethodDef CustomFloat_methods[] = {
    {"nonzero", CustomFloat_nonzero, METH_NOARGS, "True if not zero"},
    {NULL, NULL, NULL, NULL}};

static PyNumberMethods CustomFloat_as_number = {
    .nb_add = (binaryfunc)CustomFloat_add,
    .nb_subtract = (binaryfunc)CustomFloat_subtract,
    .nb_multiply = (binaryfunc)CustomFloat_multiply,
    .nb_true_divide = (binaryfunc)CustomFloat_divide,
};

static PyTypeObject CustomFloat_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "npdt.customfloat",
    .tp_doc = "custom float",
    .tp_basicsize = sizeof(CustomFloat),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT,
    .tp_new = CustomFloat_new,
    .tp_init = (initproc)CustomFloat_init,
    .tp_repr = (reprfunc)CustomFloat_repr,
    .tp_str = (reprfunc)CustomFloat_str,
    .tp_as_number = &CustomFloat_as_number,
    .tp_methods = &CustomFloat_methods};

/* ------------------------ NumPy support ---------------------------------- */

/* DType slots */

static PyObject *
CustomFloatArray_getitem(void *data, void *NPY_UNUSED(arr))
{
    CustomFloat c;
    memcpy(&c, data, sizeof(c));
    CustomFloat *cp = (CustomFloat *)CustomFloat_Type.tp_alloc(&CustomFloat_Type, 0);
    cp->value = c.value;
    return (PyObject *)cp;
}

static int
CustomFloatArray_setitem(PyObject *item, CustomFloat *cp, void *NPY_UNUSED(ap))
{
    if (CustomFloat_Check(item))
    {
        memcpy(cp, ((CustomFloat *)item), sizeof(CustomFloat));
    }
    else if (PyFloat_Check(item))
    {
        cp->value = PyFloat_AsDouble(item);
    }
    else if (PyLong_Check(item))
    {
        cp->value = PyFloat_AsDouble(item);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "unknown input to setitem");
        return -1;
    }
    return 0;
}

static PyObject *
CustomFloatArray_common_dtype(PyArray_DTypeMeta *self, PyArray_DTypeMeta *other)
{
    Py_INCREF(Py_NotImplemented);
    return (PyObject *)Py_NotImplemented;
}

static PyArray_Descr *
CustomFloatArray_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    PyErr_SetString(PyExc_ValueError, "Not implemented yet");
}

static PyArray_Descr *
CustomFloatArray_discover_descr_from_pyobject(PyArray_DTypeMeta *cls, PyObject *obj)
{
    PyErr_SetString(PyExc_ValueError, "discover_descr_from_pyobj was called. not impl");
}

/* ----------------- module def --------------------------------- */

PyMethodDef module_methods[] = {
    {0}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "customfloat",
    .m_doc = "a simple float implementation for Numpy's new DType",
    .m_size = 1};

PyMODINIT_FUNC PyInit_customfloat(void)
{
    PyObject *m = NULL;
    PyObject *numpy_str;
    PyObject *numpy;

    int experimental_dtype_version = 1;
    if (import_experimental_dtype_api(experimental_dtype_version) < 0)
    {
        return NULL;
    }

    import_array();
    if (PyErr_Occurred())
    {
        goto fail;
    }
    import_umath();
    if (PyErr_Occurred())
    {
        goto fail;
    }
    numpy_str = PyUnicode_FromString("numpy");
    if (!numpy_str)
    {
        goto fail;
    }
    numpy = PyImport_Import(numpy_str);
    Py_DecRef(numpy_str);
    if (!numpy)
    {
        goto fail;
    }

    /* Initialize customfloat type*/
    if (PyType_Ready(&CustomFloat_Type) < 0)
    {
        goto fail;
    }

    /* Initialize DTypeMeta spec*/
    PyArrayDTypeMeta_Spec spec;
    spec.typeobj = &CustomFloat_Type;
    spec.flags = 0;
    spec.baseclass = NULL;
    PyType_Slot slots[6];
    spec.slots = slots;
    slots[0].slot = NPY_DT_common_dtype;
    slots[0].pfunc = (void *)CustomFloatArray_common_dtype;
    slots[1].slot = NPY_DT_common_instance;
    slots[1].pfunc = (void *)CustomFloatArray_common_instance;
    slots[2].slot = NPY_DT_setitem;
    slots[2].pfunc = (void *)CustomFloatArray_setitem;
    slots[3].slot = NPY_DT_getitem;
    slots[3].pfunc = (void *)CustomFloatArray_getitem;
    slots[4].slot = NPY_DT_discover_descr_from_pyobject;
    slots[4].pfunc = (void *)CustomFloatArray_discover_descr_from_pyobject;
    slots[5].slot = 0;
    slots[6].pfunc = NULL;
    PyArrayMethod_Spec *castingimpls[1];
    castingimpls[0] = NULL;
    spec.casts = &castingimpls[0];

    /* Create the dtype*/
    PyArray_DTypeMeta meta;
    // PyObject *CustomFloatDType = PyArrayInitDTypeMeta_FromSpec(&meta, &spec);
    // if (!CustomFloatDType)
    // {
    //     goto fail;
    // }

    /* Create the module*/
    m = PyModule_Create(&moduledef);
    if (!m)
    {
        goto fail;
    }
    Py_IncRef(&CustomFloat_Type);
    PyModule_AddObject(m, "customfloat", (PyObject *)&CustomFloat_Type);
    return m;

fail:
    if (!PyErr_Occurred())
    {
        PyErr_SetString(PyExc_RuntimeError, "cannot load customfloat module");
    }
    if (m)
    {
        Py_DecRef(m);
        m = NULL;
    }
    return m;
}
