# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from .dmlc cimport Stream

cdef extern from "treelite/tree.h" namespace "treelite::Tree":
    cdef cppclass Node:
        Node(Tree* tree, int nid) except +
        void Serialize(Stream* fo) except +
        void Deserialize(Stream* fi) except +

cdef extern from "treelite/tree.h" namespace "treelite":
    cdef cppclass Tree:
        void Init() except +
        void Serialize(Stream* fo) except +
        void Deserialize(Stream* fi) except +
    cdef cppclass ModelParam:
        pass
    cdef cppclass Model:
        vector[Tree] trees
        int num_feature
        int num_output_group
        bool random_forest_flag
        ModelParam param
        Model() except +
        void Serialize(Stream* fo) except +
        void Deserialize(Stream* fi) except +

cdef class TreeliteModel:
    cdef unique_ptr[Model] _model
    
cdef TreeliteModel make_model(Model* handle)
cdef TreeliteModel make_stump()
