# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport move
from libc.stdint cimport uintptr_t

cdef void HandleError(int retval):
    if retval == -1:
        raise RuntimeError(TreeliteGetLastError())

cdef class TreeliteModel:
    def __cinit__(self):
        self._model = 0

    def __dealloc__(self):
        if self._model != 0:
            HandleError(TreeliteFreeModel(<ModelHandle> self._model))

cdef TreeliteModel make_model(ModelHandle handle):
    model = TreeliteModel()
    model._model = <uintptr_t> handle
    return model
