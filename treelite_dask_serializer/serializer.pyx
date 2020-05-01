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

cdef Model* make_model():
    cdef Model* model = new Model()
    cdef Tree tree
    tree.Init()
    model.trees.push_back(move(tree))
    return model

cdef string serialize(const Model* model):
    cdef string s
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    model.Serialize(strm.get())
    strm.release()
    return s

cdef Model* deserialize(string s):
    cdef Model* model = new Model()
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    model.Deserialize(strm.get())
    return model

def main():
    model = make_model()
    print('Built a tree stump with leaf output 0.0')
    s = serialize(model)
    print(f'Serialized model bytes ({len(s)} bytes) {s}')
    del model

    model2 = deserialize(s)
    print(f'Deserialized model')
    s2 = serialize(model2)
    assert s == s2
    print('Round-trip conversion preserved all bytes')
    del model2
