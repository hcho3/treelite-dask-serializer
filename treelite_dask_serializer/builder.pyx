# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from .treelite_model cimport TreeliteModel, make_model, TreeBuilderHandle, \
                             ModelBuilderHandle, ModelHandle, HandleError
from .treelite_frontend cimport *

import numpy as np

class TreeliteError(Exception):
    pass

cdef class Node:
    cdef readonly str node_type
    cdef readonly bool empty
    cdef public Tree _tree
    cdef public int _node_key

    def __init__(self):
        self.empty = True
        self._tree = None
        self._node_key = -1
        self.node_type = 'none'

    def __repr__(self):
        return f'<Node object of type {self.node_type}>'

    def set_root(self):
        if self._tree is None or self._node_key == -1:
            raise TreeliteError('This node has never been inserted into a tree; a node must ' +
                    'be inserted before it can be a root')
        HandleError(TreeliteTreeBuilderSetRootNode(self._tree._handle, self._node_key))

    def set_leaf_node(self, leaf_value):
        cdef float[::1] data
        if not self.empty:
            node_key = self._node_key if self._node_key != -1 else '_'
            raise TreeliteError('Cannot modify a non-empty node. If you meant to change type ' +
                    f'of node {node_key}, delete it first and then add an empty node with ' +
                    'the same key.')

        # check if leaf_value is a list-like object
        try:
            _ = iter(leaf_value)
            is_list = True
        except TypeError:
            is_list = False

        try:
            if is_list:
                leaf_value = [float(i) for i in leaf_value]
            else:
                leaf_value = float(leaf_value)
        except TypeError:
            raise TreeliteError('leaf_value parameter should be either a single float or a ' +
                    'list of floats')

        if self._tree is None or self._node_key == -1:
            raise TreeliteError('This node has never been inserted into a tree; a node must ' +
                    'be inserted before it can be a leaf node')

        if is_list:
            data = np.array(leaf_value, dtype=np.float32)
            HandleError(TreeliteTreeBuilderSetLeafVectorNode(self._tree._handle, self._node_key,
                &data[0], len(leaf_value)))
        else:
            HandleError(TreeliteTreeBuilderSetLeafNode(self._tree._handle, self._node_key,
                leaf_value))

        self.node_type = 'leaf'

    def set_numerical_test_node(self, feature_id, opname, threshold, default_left,
            left_child_key, right_child_key):
        if not self.empty:
            node_key = self._node_key if self._node_key != -1 else '_'
            raise ValueError('Cannot modify a non-empty node. If you meant to change type of ' +
                    f'node {node_key}, delete it first and then add an empty node with the ' +
                    'same key.')

        if self._tree is None or self._node_key == -1:
            raise TreeliteError('This node has never been inserted into a tree; a node must ' +
                    'be inserted before it can be a test node')

        # automatically create child nodes that don't exist yet
        if left_child_key not in self._tree:
            self._tree[left_child_key] = Node()
        if right_child_key not in self._tree:
            self._tree[right_child_key] = Node()
        HandleError(TreeliteTreeBuilderSetNumericalTestNode(self._tree._handle, self._node_key,
            feature_id, opname.encode('utf-8'), threshold, (1 if default_left else 0),
            left_child_key, right_child_key))
        self.empty = False
        self.node_type = 'numerical test'

    def set_categorical_test_node(self, feature_id, left_categories, default_left,
            left_child_key, right_child_key):
        cdef unsigned int[::1] data
        if not self.empty:
            node_key = self._node_key if self._node_key != -1 else '_'
            raise ValueError('Cannot modify a non-empty node. If you meant to change type of ' +
                    f'node {node_key}, delete it first and then add an empty node with the ' +
                    'same key.')

        if self._tree is None or self._node_key == -1:
            raise TreeliteError('This node has never been inserted into a tree; a node must ' +
                    'be inserted before it can be a test node')

        # automatically create child nodes that don't exist yet
        if left_child_key not in self._tree:
            self._tree[left_child_key] = Node()
        if right_child_key not in self._tree:
            self._tree[right_child_key] = Node()
        data = np.array(left_categories, dtype=np.uint32)
        HandleError(TreeliteTreeBuilderSetCategoricalTestNode(self._tree._handle, self._node_key,
            feature_id, &data[0], len(left_categories), (1 if default_left else 0), left_child_key,
            right_child_key))
        self.empty = False
        self.node_type = 'categorical test'

cdef class Tree:
    cdef TreeBuilderHandle _handle
    cdef ModelBuilderHandle _model
    cdef dict nodes

    def __cinit__(self):
        HandleError(TreeliteCreateTreeBuilder(&self._handle))

    def __dealloc__(self):
        if self._handle != NULL and self._model == NULL:
            HandleError(TreeliteDeleteTreeBuilder(self._handle))

    def __init__(self):
        self.nodes = {}

    def items(self):
        return self.nodes.items()

    def keys(self):
        return self.nodes.keys()

    def values(self):
        return self.nodes.values()

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, key):
        if key not in self.nodes:
            # implicitly create a new node
            self.__setitem__(key, Node())
        return self.nodes.__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Node):
            raise ValueError('Value must be of ModelBuilder.Node type')
        if key in self.nodes:
            raise KeyError('Nodes with duplicate keys are not allowed. If you meant to replace ' +
                    f'node {key}, delete it first and then add an empty node with the same key.')
        if not value.empty:
            raise ValueError('Can only insert an empty node')
        HandleError(TreeliteTreeBuilderCreateNode(self._handle, key))
        self.nodes.__setitem__(key, value)

        # In the node, save backlinks to the tree
        value._node_key = key
        value._tree = self

    def __delitem__(self, key):
        HandleError(TreeliteTreeBuilderDeleteNode(self._handle, key))
        self.nodes.__delitem__(key)

    def __iter__(self):
        return self.nodes.__iter__()

    def __repr__(self):
        return (f'<Tree object containing {len(self.nodes)} nodes>\n' +
                dict(self).__repr__())

cdef class ModelBuilder:
    cdef ModelBuilderHandle _handle
    cdef list trees

    def __cinit__(self):
        self._handle = NULL

    def __dealloc__(self):
        if self._handle != NULL:
            HandleError(TreeliteDeleteModelBuilder(self._handle))

    def __init__(self, num_feature, num_output_group=1, random_forest=False, **kwargs):
        if not isinstance(num_feature, int):
            raise ValueError('num_feature must be of int type')
        if num_feature <= 0:
            raise ValueError('num_feature must be strictly positive')
        if not isinstance(num_output_group, int):
            raise ValueError('num_output_group must be of int type')
        if num_output_group <= 0:
            raise ValueError('num_output_group must be strictly positive')
        HandleError(TreeliteCreateModelBuilder(num_feature, num_output_group, random_forest,
            &self._handle))
        for key, value in kwargs.items():
            if not isinstance(value, (str,)):
                value = str(value)
            HandleError(TreeliteModelBuilderSetModelParam(self._handle, key.encode('utf-8'),
                value.encode('utf-8')))
        self.trees = []

    def insert(self, index, Tree tree):
        if not isinstance(index, int):
            raise ValueError('index must be of int type')
        if index < 0 or index > len(self):
            raise ValueError('index out of bounds')
        HandleError(TreeliteModelBuilderInsertTree(self._handle, tree._handle, index))

        HandleError(TreeliteDeleteTreeBuilder(tree._handle))
        HandleError(TreeliteModelBuilderGetTree(self._handle, index, &tree._handle))
        tree._model = self._handle
        self.trees.insert(index, tree)

    def append(self, Tree tree):
        self.insert(len(self), tree)

    def commit(self):
        cdef ModelHandle handle
        HandleError(TreeliteModelBuilderCommitModel(self._handle, &handle))
        return make_model(handle)

    ### Implement list semantics whenever applicable
    def __len__(self):
        return len(self.trees)

    def __getitem__(self, index):
        return self.trees.__getitem__(index)

    def __delitem__(self, index):
        HandleError(TreeliteModelBuilderDeleteTree(self._handle, index))
        self.trees.__delitem__(index)

    def __iter__(self):
        return self.trees.__iter__()

    def __reversed__(self):
        return self.trees.__reversed__()

    def __repr__(self):
        return f'<treelite.ModelBuilder object storing {len(self.trees)} decision trees>\n'

