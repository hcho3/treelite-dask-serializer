# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from .dmlc cimport Stream

cdef extern from "treelite/tree.h" namespace "treelite":
    cdef cppclass PyBufferFrame:
        void* buf
        char* format
        size_t itemsize
        size_t nitem
    cdef cppclass Tree:
        void Init() except +
        void ReferenceSerialize(Stream* fo) except +
    cdef cppclass ModelParam:
        pass
    cdef cppclass Model:
        vector[Tree] trees
        int num_feature
        int num_output_group
        bool random_forest_flag
        ModelParam param
        Model() except +
        void ReferenceSerialize(Stream* fo) except +
        vector[PyBufferFrame] GetPyBuffer() except +
        void InitFromPyBuffer(vector[PyBufferFrame] frames) except +

cdef class TreeliteModel:
    cdef unique_ptr[Model] _model
    
cdef TreeliteModel make_model(Model* handle)
cdef TreeliteModel make_stump()
