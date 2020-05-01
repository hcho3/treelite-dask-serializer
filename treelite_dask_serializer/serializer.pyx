# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from libcpp.utility cimport move

cdef extern from "dmlc/io.h" namespace "dmlc":
    cdef cppclass Stream:
        pass

cdef extern from "dmlc/memory_io.h" namespace "dmlc":
    cdef cppclass MemoryStringStream:
        MemoryStringStream(string* p_buffer) except +
        size_t Read(void* ptr, size_t size) except +
        void Write(const void* ptr, size_t size) except +

cdef extern from "treelite/tree.h" namespace "treelite::Tree":
    cdef cppclass Node:
        Node(Tree* tree, int nid) except +
        void set_leaf(float value) except +
        void Serialize(Stream* fo) except +
        void Deserialize(Stream* fi) except +

cdef extern from "treelite/tree.h" namespace "treelite":
    cdef cppclass Tree:
        void Init() except +
        Node& operator[](int nid) except +
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
    cdef Model* _model

    def __cinit__(self):
        self._model = new Model()

    def __dealloc__(self):
        if self._model != NULL:
            del self._model

    cdef string serialize(self):
        cdef string s
        cdef unique_ptr[Stream] strm
        strm.reset(new MemoryStringStream(&s))
        self._model.Serialize(strm.get())
        strm.release()
        return s

    @staticmethod
    cdef TreeliteModel deserialize(string s):
        model = TreeliteModel()
        cdef unique_ptr[Stream] strm
        strm.reset(new MemoryStringStream(&s))
        model._model.Deserialize(strm.get())
        strm.release()
        return model

    @staticmethod
    cdef TreeliteModel make_stump():
        model = TreeliteModel()
        cdef Tree tree
        tree.Init()
        model._model.trees.push_back(move(tree))
        return model

def main():
    model = TreeliteModel.make_stump()
    print('Built a tree stump with leaf output 0.0')
    s = model.serialize()
    print(f'Serialized model bytes ({len(s)} bytes) {s}')
    del model

    model2 = TreeliteModel.deserialize(s)
    print(f'Deserialized model')
    s2 = model2.serialize()
    assert s == s2, f'len(s2) = {len(s2)}'
    print('Round-trip conversion preserved all bytes')
    del model2
