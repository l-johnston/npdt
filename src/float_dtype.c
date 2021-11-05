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
PyObject *FloatSingleton = NULL;

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
        .tp_name = "npdt.float_dtype.FloatDType",
    .tp_methods = float_methods,
    .tp_new = float_new,
    .tp_repr = (reprfunc)float_repr,
    .tp_str = (reprfunc)float_repr,
    .tp_basicsize = sizeof(PyArray_FloatDescr),
    // .tp_flags = Py_TPFLAGS_HEAPTYPE,
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

static NPY_CASTING
float_to_float_resolve_descriptors(PyObject *NPY_UNUSED(method), PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]), PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2])
{
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(loop_descrs[0]);
    if (given_descrs[1] == NULL)
    {
        loop_descrs[1] = given_descrs[0];
    }
    else
    {
        loop_descrs[1] = given_descrs[1];
    }
    Py_INCREF(loop_descrs[1]);
    return NPY_SAME_KIND_CASTING;
}

static NPY_CASTING
pyfloat_to_from_float_resolve_descriptors(PyObject *NPY_UNUSED(method), PyArray_DTypeMeta *dtypes[2], PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2])
{
    loop_descrs[0] = given_descrs[0];
    if (loop_descrs[0] == NULL)
    {
        return -1;
    }
    Py_INCREF(loop_descrs[0]);
    loop_descrs[1] = given_descrs[1];
    if (loop_descrs[1] == NULL)
    {
        loop_descrs[1] = FloatSingleton;
    }
    Py_INCREF(loop_descrs[1]);
    return NPY_NO_CASTING | _NPY_CAST_IS_VIEW;
}

cast_int_to_float_aligned(PyArrayMethod_Context *context,
                          char *const data[], npy_intp const dimensions[],
                          npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++)
    {
        *(double *)out = *(long int *)in;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

cast_float_to_int_aligned(PyArrayMethod_Context *context,
                          char *const data[], npy_intp const dimensions[],
                          npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    for (npy_intp i = 0; i < N; i++)
    {
        *(long int *)out = *(double *)in;
        in += strides[0];
        out += strides[1];
    }
    return 0;
}

static NPY_CASTING
pyint_to_from_float_resolve_descriptors(PyObject *NPY_UNUSED(method), PyArray_DTypeMeta *dtypes[2], PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2])
{
    loop_descrs[0] = given_descrs[0];
    if (loop_descrs[0] == NULL)
    {
        return -1;
    }
    Py_INCREF(loop_descrs[0]);
    loop_descrs[1] = given_descrs[1];
    if (loop_descrs[1] == NULL)
    {
        loop_descrs[1] = FloatSingleton;
    }
    Py_INCREF(loop_descrs[1]);
    return NPY_UNSAFE_CASTING;
}

static int
multipy_floats(PyArrayMethod_Context *context, char **data, npy_intp *dimensions, npy_intp *strides, void *userdata)
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    for (npy_intp i = 0; i < N; i++)
    {
        *(double *)out = *(double *)in1 * *(double *)in2;
        in1 += strides[0];
        in2 += strides[1];
        out += strides[2];
    }
    return 0;
}

static NPY_CASTING
multiply_floats_resolve_descriptors(PyObject *method, PyObject *dtypes[3], PyObject *given_descrs[3], PyObject *loop_descrs[3])
{
    loop_descrs[2] = given_descrs[0];
    if (loop_descrs[2] == 0)
    {
        return -1;
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];
    return NPY_NO_CASTING;
}

static int
promote_to_float(PyUFuncObject *ufunc, PyObject *dtypes[3], PyObject *signature[3], PyObject *new_dtypes[3])
{
    for (int i = 0; i < 3; i++)
    {
        PyObject *new = &PyArray_FloatDType;
        if (signature[i] != NULL)
        {
            new = signature[i];
        }
        Py_INCREF(new);
        new_dtypes[i] = new;
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
    PyObject *ufunc;

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
    ufunc = PyObject_GetAttrString(numpy, "multiply");
    Py_DECREF(numpy);
    if (ufunc == NULL)
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
    spec.flags = NPY_DT_PARAMETRIC;
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
    PyArrayMethod_Spec *castingimpls[6] = {NULL};
    spec.casts = castingimpls;
    PyArray_DTypeMeta *dtypes[2] = {&PyArray_FloatDType, &PyArray_FloatDType};
    PyType_Slot float_to_float_slots[4] = {{0, NULL}};
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
    float_to_float_slots[2].slot = NPY_METH_resolve_descriptors;
    float_to_float_slots[2].pfunc = &float_to_float_resolve_descriptors;
    castingimpls[0] = &float_to_float_spec;
    PyType_Slot pyfloat_to_float_slots[3] = {{0, NULL}};
    PyArray_DTypeMeta *double_DType = PyArray_DTypeFromTypeNum(NPY_DOUBLE);
    Py_DecRef(double_DType);
    PyArray_DTypeMeta *int_DType = PyArray_DTypeFromTypeNum(NPY_INT64);
    Py_DecRef(int_DType);
    PyArray_DTypeMeta *pyfloat_to_float_dtypes[2] = {double_DType, &PyArray_FloatDType};
    PyArrayMethod_Spec pyfloat_to_float_spec = {
        .name = "pyfloat_to_float_cast",
        .nin = 1,
        .nout = 1,
        .dtypes = pyfloat_to_float_dtypes,
        .casting = NPY_SAFE_CASTING,
        .slots = pyfloat_to_float_slots,
        .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
    };
    pyfloat_to_float_slots[0].slot = NPY_METH_strided_loop;
    pyfloat_to_float_slots[0].pfunc = &cast_float_to_float_aligned;
    pyfloat_to_float_slots[1].slot = NPY_METH_resolve_descriptors;
    pyfloat_to_float_slots[1].pfunc = &pyfloat_to_from_float_resolve_descriptors;
    castingimpls[1] = &pyfloat_to_float_spec;
    PyType_Slot float_to_pyfloat_slots[3] = {{0, NULL}};
    PyArray_DTypeMeta *float_to_pyfloat_dtypes[2] = {&PyArray_FloatDType, double_DType};
    PyArrayMethod_Spec float_to_pyfloat_spec = {
        .name = "float_to_pyfloat_cast",
        .nin = 1,
        .nout = 1,
        .dtypes = float_to_pyfloat_dtypes,
        .casting = NPY_SAFE_CASTING,
        .slots = float_to_pyfloat_slots,
        .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
    };
    float_to_pyfloat_slots[0].slot = NPY_METH_strided_loop;
    float_to_pyfloat_slots[0].pfunc = &cast_float_to_float_aligned;
    float_to_pyfloat_slots[1].slot = NPY_METH_resolve_descriptors;
    float_to_pyfloat_slots[1].pfunc = &pyfloat_to_from_float_resolve_descriptors;
    castingimpls[2] = &float_to_pyfloat_spec;
    PyType_Slot pyint_to_float_slots[3] = {{0, NULL}};
    PyArray_DTypeMeta *pyint_to_float_dtypes[2] = {int_DType, &PyArray_FloatDType};
    PyArrayMethod_Spec pyint_to_float_spec = {
        .name = "pyint_to_float_cast",
        .nin = 1,
        .nout = 1,
        .dtypes = pyint_to_float_dtypes,
        .casting = NPY_SAFE_CASTING,
        .slots = pyint_to_float_slots,
        .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
    };
    pyint_to_float_slots[0].slot = NPY_METH_strided_loop;
    pyint_to_float_slots[0].pfunc = &cast_int_to_float_aligned;
    pyint_to_float_slots[1].slot = NPY_METH_resolve_descriptors;
    pyint_to_float_slots[1].pfunc = &pyint_to_from_float_resolve_descriptors;
    castingimpls[3] = &pyint_to_float_spec;
    PyType_Slot float_to_pyint_slots[3] = {{0, NULL}};
    PyArray_DTypeMeta *float_to_pyint_dtypes[2] = {&PyArray_FloatDType, int_DType};
    PyArrayMethod_Spec float_to_pyint_spec = {
        .name = "float_to_pyint_cast",
        .nin = 1,
        .nout = 1,
        .dtypes = float_to_pyint_dtypes,
        .casting = NPY_SAFE_CASTING,
        .slots = float_to_pyint_slots,
        .flags = NPY_METH_NO_FLOATINGPOINT_ERRORS,
    };
    float_to_pyint_slots[0].slot = NPY_METH_strided_loop;
    float_to_pyint_slots[0].pfunc = &cast_float_to_int_aligned;
    float_to_pyint_slots[1].slot = NPY_METH_resolve_descriptors;
    float_to_pyint_slots[1].pfunc = &pyint_to_from_float_resolve_descriptors;
    castingimpls[5] = &pyint_to_float_spec;
    if (PyArrayInitDTypeMeta_FromSpec(&PyArray_FloatDType, &spec) < 0)
    {
        goto fail;
    }
    PyArray_DTypeMeta *multiply_floats_dtypes[3] = {&PyArray_FloatDType, &PyArray_FloatDType, &PyArray_FloatDType};
    PyType_Slot multiply_floats_slots[3] = {{0, NULL}};
    PyArrayMethod_Spec multiply_floats_spec = {
        .nin = 2,
        .nout = 1,
        .dtypes = multiply_floats_dtypes,
        .slots = multiply_floats_slots,
        .name = "multiply_floats",
        .casting = NPY_NO_CASTING,
        .flags = 0,
    };
    multiply_floats_slots[0].slot = NPY_METH_strided_loop;
    multiply_floats_slots[0].pfunc = &multipy_floats;
    multiply_floats_slots[1].slot = NPY_METH_resolve_descriptors;
    multiply_floats_slots[1].pfunc = &multiply_floats_resolve_descriptors;
    if (PyUFunc_AddLoopFromSpec(ufunc, &multiply_floats_spec) < 0)
    {
        goto fail;
    }
    PyObject *promoter_dtypes = Py_BuildValue("(OOO)", double_DType, &PyArray_FloatDType, &PyArray_FloatDType);
    PyObject *promoter = PyCapsule_New(&promote_to_float, "numpy._ufunc_promoter", NULL);
    if (promoter == NULL)
    {
        goto fail;
    }
    if (PyUFunc_AddPromoter(ufunc, promoter_dtypes, promoter) < 0)
    {
        goto fail;
    }
    promoter_dtypes = Py_BuildValue("(OOO)", &PyArray_FloatDType, double_DType, &PyArray_FloatDType);
    if (PyUFunc_AddPromoter(ufunc, promoter_dtypes, promoter) < 0)
    {
        goto fail;
    }
    promoter_dtypes = Py_BuildValue("(OOO)", int_DType, &PyArray_FloatDType, &PyArray_FloatDType);
    if (PyUFunc_AddPromoter(ufunc, promoter_dtypes, promoter) < 0)
    {
        goto fail;
    }
    promoter_dtypes = Py_BuildValue("(OOO)", &PyArray_FloatDType, int_DType, &PyArray_FloatDType);
    if (PyUFunc_AddPromoter(ufunc, promoter_dtypes, promoter) < 0)
    {
        goto fail;
    }
    FloatSingleton = PyObject_CallNoArgs((PyObject *)&PyArray_FloatDType);
    if (FloatSingleton == NULL)
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
