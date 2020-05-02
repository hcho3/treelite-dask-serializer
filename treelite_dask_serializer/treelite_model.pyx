# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.utility cimport move

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
