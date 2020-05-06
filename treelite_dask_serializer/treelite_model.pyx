# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport move
from cython.operator cimport dereference as deref, preincrement as inc

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

cdef class TreeliteModel:
    def __cinit__(self):
        self._model.reset(new Model())

    def __dealloc__(self):
        self._model.release()

cdef TreeliteModel make_model(Model* handle):
    model = TreeliteModel()
    model._model.reset(handle)
    return model

cdef TreeliteModel make_stump():
    model = TreeliteModel()
    cdef Tree tree
    tree.Init()
    model._model.get().trees.push_back(move(tree))
    return model

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

def get_frames(model):
    return _get_frames(model)
