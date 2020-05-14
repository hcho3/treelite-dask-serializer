# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from .dmlc cimport Stream

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    ctypedef void* TreeBuilderHandle
    ctypedef void* ModelBuilderHandle
    cdef const char* TreeliteGetLastError()
    cdef int TreeliteFreeModel(ModelHandle handle)
cdef extern from "treelite/tree.h" namespace "treelite":
    cdef struct PyBufferFrame:
        void* buf
        char* format
        size_t itemsize
        size_t nitem

cdef class TreeliteModel:
    cdef uintptr_t _model
    
cdef TreeliteModel make_model(ModelHandle handle)
cdef void HandleError(int retval)
