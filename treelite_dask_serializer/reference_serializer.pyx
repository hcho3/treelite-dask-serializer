# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3
"""Reference serializer implementation"""

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from .treelite_model cimport TreeliteModel, make_model, make_stump
from .dmlc cimport Stream, MemoryStringStream

cdef string _reference_serialize(TreeliteModel model) except *:
    cdef string s
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    model._model.get().ReferenceSerialize(strm.get())
    strm.reset()
    return s

def treelite2bytes(model : TreeliteModel) -> bytes:
    return _reference_serialize(model)
