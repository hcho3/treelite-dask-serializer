# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from .treelite_model cimport ModelHandle, TreeBuilderHandle, ModelBuilderHandle

cdef extern from "treelite/c_api.h":
    cdef int TreeliteCreateTreeBuilder(TreeBuilderHandle* out)
    cdef int TreeliteDeleteTreeBuilder(TreeBuilderHandle handle)
    cdef int TreeliteTreeBuilderCreateNode(TreeBuilderHandle handle, int node_key)
    cdef int TreeliteTreeBuilderDeleteNode(TreeBuilderHandle handle, int node_key)
    cdef int TreeliteTreeBuilderSetRootNode(TreeBuilderHandle handle, int node_key)
    cdef int TreeliteTreeBuilderSetNumericalTestNode(TreeBuilderHandle handle, int node_key,
            unsigned feature_id, const char* opname, float threshold, int default_left,
            int left_child_key, int right_child_key)
    cdef int TreeliteTreeBuilderSetCategoricalTestNode(TreeBuilderHandle handle, int node_key,
            unsigned feature_id, const unsigned int* left_categories, size_t left_categories_len,
            int default_left, int left_child_key, int right_child_key)
    cdef int TreeliteTreeBuilderSetLeafNode(TreeBuilderHandle handle, int node_key,
            float leaf_value)
    cdef int TreeliteTreeBuilderSetLeafVectorNode(TreeBuilderHandle handle, int node_key,
            const float* leaf_vector, size_t leaf_vector_len)
    cdef int TreeliteCreateModelBuilder(int num_feature, int num_output_group,
            int random_forest_flag, ModelBuilderHandle* out)
    cdef int TreeliteDeleteModelBuilder(ModelBuilderHandle handle)
    cdef int TreeliteModelBuilderSetModelParam(ModelBuilderHandle handle, const char* name,
            const char* value)
    cdef int TreeliteModelBuilderInsertTree(ModelBuilderHandle handle,
            TreeBuilderHandle tree_builder_handle, int index) 
    cdef int TreeliteModelBuilderDeleteTree(ModelBuilderHandle handle, int index)
    cdef int TreeliteModelBuilderGetTree(ModelBuilderHandle handle, int index,
            TreeBuilderHandle *out)
    cdef int TreeliteModelBuilderCommitModel(ModelBuilderHandle handle, ModelHandle* out)
