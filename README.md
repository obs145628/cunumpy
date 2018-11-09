# cunumpy

Default implementation of np.dot() for matrix multiplication is cblas. 
This project remplaces the call to cblas by a call to cublas. 
It's only for experiment purposes, and not for practical usages.

## Build

Tested with Cuda 10 and Python 3.6
```shell
cd src
python setup.py build
```

## Examples

```
cd tests
LD_PRELOAD=/path/to/cunumpy/src/build/lib.linux-x86_64-3.6/pyhack.cpython-36m-x86_64-linux-gnu.so PYTHONPATH=../src/build/lib.linux-x86_64-3.6/ python matmul-n.py 
```