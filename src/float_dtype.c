#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <numpy/ufuncobject.h>
#include <numpy/experimental_dtype_api.h>
#include <stddef.h>

#define _ALIGN(type) offsetof( \
    struct                     \
    {                          \
        char c;                \
        type v;                \
    },                         \
    v)

#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))
static NPY_INLINE PyArray_DTypeMeta *
PyArray_DTypeFromTypeNum(int typenum)
{
    PyArray_Descr *descr = PyArray_DescrFromType(typenum);
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);
    Py_INCREF(dtype);
    Py_DECREF(descr);
    return dtype;
}

typedef struct
{
    PyArray_Descr base;
} PyArray_FloatDescr;

static PyArray_DTypeMeta PyArray_FloatDType;

static int
float_is_known_scalar_type(PyArray_DTypeMeta *NPY_UNUSED(cls), PyTypeObject *type)
{
    if (type == &PyFloat_Type)
    {
        return 1;
    }
    return 0;
}

static PyArray_Descr *
float_default_descr(PyArray_DTypeMeta *NPY_UNUSED(cls))
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyArray_Descr *
float_discover_from_pyobject(PyArray_DTypeMeta *cls, PyObject *NPY_UNUSED(obj))
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyArray_DTypeMeta *
float_common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyArray_Descr *
float_common_instance(PyArray_Descr *descr1, PyArray_Descr *descr2)
{
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
}

static PyObject *
float_getitem(PyArray_FloatDescr *self, char *ptr)
{
    double value;
    memcpy((void *)&value, ptr, sizeof(double));
    return PyFloat_FromDouble(value);
}

static int
float_setitem(PyArray_FloatDescr *self, PyObject *obj, char *ptr)
{
    if (!PyFloat_CheckExact(obj))
    {
        PyErr_SetString(PyExc_NotImplementedError, "Currently only supports float");
        return -1;
    }

    double value = PyFloat_AsDouble(obj);
    memcpy(ptr, (void *)&value, sizeof(double));
    return 0;
}

PyMethodDef float_methods[] = {
    {NULL, NULL, 0, NULL},
};

static PyObject *
float_new(struct PyArray_FloatDType *cls, PyObject *args, PyObject *kwargs)
{
    PyArray_Descr *new = (PyArray_Descr *)(PyArrayDescr_Type.tp_new(cls, args, kwargs));
    if (new == NULL)
    {
        return NULL;
    }
    new->elsize = sizeof(double);
    new->alignment = _ALIGN(double);
    return new;
}

static PyObject *
float_repr(PyArray_FloatDescr *self)
{
    PyObject *res = PyUnicode_FromString("FloatDType");
    return res;
}

static PyArray_DTypeMeta PyArray_FloatDType = {{{
    PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "FloatDType",
    .tp_methods = float_methods,
    .tp_new = float_new,
    .tp_repr = (reprfunc)float_repr,
    .tp_str = (reprfunc)float_repr,
    .tp_basicsize = sizeof(PyArray_FloatDescr),
    .tp_flags = Py_TPFLAGS_HEAPTYPE,
}}};

static int
cast_float_to_float_unaligned(PyArrayMethod_Context *context,
                              char *const data[], npy_intp const dimensions[],
                              npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++)
    {
        memcpy(out, in, sizeof(double));
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

static int
cast_float_to_float_aligned(PyArrayMethod_Context *context,
                            char *const data[], npy_intp const dimensions[],
                            npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++)
    {
        *(double *)out = *(double *)in;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

PyObject *
get_anint(PyObject *self, PyObject *args)
{
    return PyLong_FromLong(2);
}

PyMethodDef module_methods[] = {
    {"get_anint", &get_anint, METH_NOARGS, "get an int"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "float_dtype",
    .m_doc = "a simple float implementation for Numpy's new DType",
    .m_size = 1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC
PyInit_float_dtype(void)
{
    PyObject *m = NULL;
    PyObject *numpy_str;
    PyObject *numpy;

    int experimental_dtype_version = 2;
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
    Py_SET_TYPE(&PyArray_FloatDType, &PyArrayDTypeMeta_Type);
    ((PyTypeObject *)&PyArray_FloatDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&PyArray_FloatDType) < 0)
    {
        goto fail;
    }
    PyArrayDTypeMeta_Spec spec;
    spec.typeobj = &PyArray_FloatDType;
    spec.flags = 0;
    spec.baseclass = NULL;
    PyType_Slot slots[8] = {{0, NULL}};
    spec.slots = slots;
    slots[0].slot = NPY_DT_common_dtype;
    slots[0].pfunc = &float_common_dtype;
    slots[1].slot = NPY_DT_common_instance;
    slots[1].pfunc = &float_common_instance;
    slots[2].slot = NPY_DT_default_descr;
    slots[2].pfunc = &float_default_descr;
    slots[3].slot = NPY_DT_discover_descr_from_pyobject;
    slots[3].pfunc = &float_discover_from_pyobject;
    slots[4].slot = NPY_DT_getitem;
    slots[4].pfunc = &float_getitem;
    slots[5].slot = NPY_DT_setitem;
    slots[5].pfunc = &float_setitem;
    PyArrayMethod_Spec *castingimpls[2];
    spec.casts = castingimpls;
    PyArray_DTypeMeta *dtypes[2] = {&PyArray_FloatDType, &PyArray_FloatDType};
    PyType_Slot float_to_float_slots[3] = {{0, NULL}};
    PyArrayMethod_Spec float_to_float_spec = {
        .name = "float_to_float_cast",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = dtypes,
        .casting = NPY_SAME_KIND_CASTING,
        .slots = float_to_float_slots,
    };
    float_to_float_slots[0].slot = NPY_METH_strided_loop;
    float_to_float_slots[0].pfunc = &cast_float_to_float_aligned;
    float_to_float_slots[1].slot = NPY_METH_unaligned_strided_loop;
    float_to_float_slots[1].pfunc = &cast_float_to_float_unaligned;
    castingimpls[0] = &float_to_float_spec;
    PyType_Slot pyfloat_to_from_float_slots[2] = {{0, NULL}};
    PyArray_DTypeMeta *double_DType = PyArray_DTypeFromTypeNum(NPY_DOUBLE);
    Py_DecRef(double_DType);
    PyArray_DTypeMeta *pyfloat_to_from_float_dtypes[2] = {double_DType, &PyArray_FloatDType};
    PyArrayMethod_Spec pyfloat_to_from_float_spec = {
        .name = "pyfloat_to_from_float_cast",
        .nin = 1,
        .nout = 1,
        .dtypes = pyfloat_to_from_float_dtypes,
        .casting = NPY_SAFE_CASTING,
        .slots = pyfloat_to_from_float_slots,
        .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
    };
    pyfloat_to_from_float_slots[0].slot = NPY_METH_strided_loop;
    pyfloat_to_from_float_slots[0].pfunc = &cast_float_to_float_aligned;
    castingimpls[1] = &pyfloat_to_from_float_spec;
    if (PyArrayInitDTypeMeta_FromSpec(&PyArray_FloatDType, &spec) < 0)
    {
        goto fail;
    }
    m = PyModule_Create(&moduledef);
    if (!m)
    {
        goto fail;
    }
    Py_IncRef(&PyArray_FloatDType);
    PyModule_AddObject(m, "FloatDType", (PyObject *)&PyArray_FloatDType);
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
