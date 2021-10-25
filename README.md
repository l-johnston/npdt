# npdt - Experiment with Numpy's new DType

The first experiment is to create a simple floating point DType called `customfloat`
to work through the mechanics of the new API without additional complexity.

As of 2021-10-24:
- Linux
- Python 3.10
- clone numpy repo
- clone npdt repo
- create virtual env
- install numpy, etc.
- pip install -e .
- export NUMPY_EXPERIMENTAL_DTYPE_API=1

Usage
```
>>> from npdt import customfloat
>>> customfloat(2)
2
>>> 2*_
4
>>> 2+_
6
```

At the moment, having trouble with `PyArrayDTypeMeta_Spec`, `PyArray_DTypeMeta` and
`PyArrayDTypeMeta_Type`.
