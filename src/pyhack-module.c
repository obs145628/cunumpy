#include <Python.h>
#include "wrapper.h"

static PyObject* pyhack_enable_wrapper(PyObject* self, PyObject* args)
{
  g_use_wrapper = 1;
  return Py_None;
}

static struct PyMethodDef methods[] =
{
  {"enable_wrapper", pyhack_enable_wrapper, METH_NOARGS, "Replace methods by wrapper"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def =
{
  PyModuleDef_HEAD_INIT,
  "pyhack",
  "...",
  -1,
  methods
};

PyMODINIT_FUNC PyInit_pyhack(void)
{
  return PyModule_Create(&module_def);
}
