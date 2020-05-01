# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref

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

cdef Tree* make_model():
    cdef Tree* tree = new Tree()
    tree.Init()
    return tree

cdef string serialize(Tree* tree):
    cdef string s
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    tree.Serialize(strm.get())
    strm.release()
    return s

cdef Tree* deserialize(string s):
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    cdef Tree* tree = new Tree()
    tree.Deserialize(strm.get())
    return tree

def main():
    model = make_model()
    print('Built a tree stump with leaf output 0.0')
    s = serialize(model)
    print(f'Serialized model bytes ({len(s)} bytes) {s}')
    model2 = deserialize(s)
    print(f'Deserialized model')
    s2 = serialize(model2)
    assert s == s2
    print('Round-trip conversion preserved all bytes')
    del model
    del model2
