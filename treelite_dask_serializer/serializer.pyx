# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from .treelite_model cimport TreeliteModel, make_model, make_stump
from .dmlc cimport Stream, MemoryStringStream

cdef string _serialize(TreeliteModel model) except *:
    cdef string s
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    model._model.get().Serialize(strm.get())
    strm.reset()
    return s

cdef TreeliteModel _deserialize(string s):
    model = TreeliteModel()
    cdef unique_ptr[Stream] strm
    strm.reset(new MemoryStringStream(&s))
    model._model.get().Deserialize(strm.get())
    strm.reset()
    return model

def serialize(model):
    return _serialize(model)

def deserialize(s):
    return _deserialize(s)
