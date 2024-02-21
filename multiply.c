#include <Python.h>

static PyObject* multiply(PyObject* self, PyObject* args) {
    double a, b;
    if (!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }
    return Py_BuildValue("d", a + b);
}

static PyMethodDef methods[] = {
    {"multiply", multiply, METH_VARARGS, "Multiply two numbers"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "multiply",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_multiply(void) {
    return PyModule_Create(&module);
}
