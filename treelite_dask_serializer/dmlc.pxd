# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.string cimport string

cdef extern from "dmlc/io.h" namespace "dmlc":
    cdef cppclass Stream:
        pass

cdef extern from "dmlc/memory_io.h" namespace "dmlc":
    cdef cppclass MemoryStringStream:
        MemoryStringStream(string* p_buffer) except +
        size_t Read(void* ptr, size_t size) except +
        void Write(const void* ptr, size_t size) except +

