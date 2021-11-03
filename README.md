# npdt - Experiment with Numpy's new DType

The first experiment is to create a simple floating point DType called `FloatDType`
to work through the mechanics of the new API without additional complexity.

As of 2021-10-24:
- Linux
- Python 3.10
- clone numpy repo
- clone npdt repo
- create virtual env
- install numpy, etc.
- python setup.py build develop
- export NUMPY_EXPERIMENTAL_DTYPE_API=1

Usage
```
>>> from npdt import FloatDType
>>> dt = FloatDType()
>>> import numpy as np
>>> np.array([1.0, 2.0], dtype=dt)
array([1.0, 2.0], dtype=FloatDType)
```
