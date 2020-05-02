# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uint32_t
from .treelite_model cimport Model

cdef extern from "treelite/frontend.h" namespace "treelite::frontend":
    cdef cppclass TreeBuilder:
        TreeBuilder() except +
        bool CreateNode(int node_key) except +
        bool DeleteNode(int node_key) except +
        bool SetRootNode(int node_key) except +
        bool SetNumericalTestNode(int node_key, unsigned feature_id, const char* op,
                float threshold, bool default_left, int left_child_key,
                int right_child_key) except +
        bool SetCategoricalTestNode(int node_key, unsigned feature_id,
                const vector[uint32_t]& left_categories, bool default_left, int left_child_key,
                int right_child_key) except +
        bool SetLeafNode(int node_key, float leaf_value) except +
        bool SetLeafVectorNode(int node_key, const vector[float]& leaf_vector) except +
    cdef cppclass ModelBuilder:
        ModelBuilder(int num_feature, int num_output_group, bool random_forest_flag) except +
        void SetModelParam(const char* name, const char* value) except +
        int InsertTree(TreeBuilder* tree_builder, int index) except +
        TreeBuilder* GetTree(int index) except +
        bool DeleteTree(int index) except +
        bool CommitModel(Model* out_model) except +
