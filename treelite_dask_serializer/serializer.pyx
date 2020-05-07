# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from typing import List, Union
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc
from cpython.object cimport PyObject
from .treelite_model cimport TreeliteModel
from .treelite_model cimport PyBufferFrame

cdef extern from "Python.h":
    Py_buffer* PyMemoryView_GET_BUFFER(PyObject* mview)

cdef class PyBufferFrameWrapper:
    cdef PyBufferFrame _handle
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_ssize_t itemsize = self._handle.itemsize

        self.shape[0] = self._handle.nitem
        self.strides[0] = itemsize

        buffer.buf = self._handle.buf
        buffer.format = self._handle.format
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self._handle.nitem * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef PyBufferFrameWrapper MakePyBufferFrameWrapper(PyBufferFrame handle):
    cdef PyBufferFrameWrapper wrapper = PyBufferFrameWrapper()
    wrapper._handle = handle
    return wrapper

cdef list _get_frames(TreeliteModel model):
    frames = []
    cdef vector[PyBufferFrame] interface = model._model.get().GetPyBuffer()
    cdef vector[PyBufferFrame].iterator it = interface.begin()
    while it != interface.end():
        v = deref(it)
        w = MakePyBufferFrameWrapper(v)
        frames.append(memoryview(w))
        inc(it)
    return frames

cdef TreeliteModel _init_from_frames(vector[PyBufferFrame] frames):
    model = TreeliteModel()
    model._model.get().InitFromPyBuffer(frames)
    return model

def get_frames(model : TreeliteModel) -> List[memoryview]:
    return _get_frames(model)

def init_from_frames(frames : List[Union[bytes, memoryview]], format_str: List[str], itemsize: List[int]) -> TreeliteModel:
    cdef vector[PyBufferFrame] cpp_frames
    cdef Py_buffer* buf
    cdef PyBufferFrame cpp_frame
    for i, frame in enumerate(frames):
        if isinstance(frame, memoryview):
            buf = PyMemoryView_GET_BUFFER(<PyObject*>frame)
            cpp_frame.buf = buf.buf
            cpp_frame.format = buf.format
            cpp_frame.itemsize = buf.itemsize
            cpp_frame.nitem = buf.shape[0]
        else:
            cpp_frame.buf = <char*>(frame)
            cpp_frame.format = format_str[i]
            cpp_frame.itemsize = itemsize[i]
            cpp_frame.nitem = len(frame) // itemsize[i]
        cpp_frames.push_back(cpp_frame)
    return _init_from_frames(cpp_frames)
