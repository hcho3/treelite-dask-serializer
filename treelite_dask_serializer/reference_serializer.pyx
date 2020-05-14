# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
"""Reference serializer implementation"""

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from .treelite_model cimport TreeliteModel
from .dmlc cimport Stream, MemoryStringStream

cdef extern from "treelite/tree.h" namespace "treelite":
    cdef cppclass Model:
        void ReferenceSerialize(Stream* fo) except +

cdef string _reference_serialize(TreeliteModel model) except *:
    cdef string s
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    cdef Model* ptr = <Model*>model._model
    ptr.ReferenceSerialize(strm.get())
    strm.reset()
    return s

def treelite2bytes(model : TreeliteModel) -> bytes:
    return _reference_serialize(model)
