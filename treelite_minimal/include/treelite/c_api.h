/*!
 * Copyright (c) 2020 by Contributors
 * \file c_api.h
 * \author Hyunsu Cho
 * \brief C API of Treelite, used for interfacing with other languages
 *        This header is excluded from the runtime
 */

/* Note: Make sure to use slash-asterisk form of comments in this file
   (like this one). Do not use double-slash (//). */

#ifndef TREELITE_C_API_H_
#define TREELITE_C_API_H_

#include "c_api_common.h"

/*!
 * \addtogroup opaque_handles
 * opaque handles
 * \{
 */
/*! \brief handle to a decision tree ensemble model */
typedef void* ModelHandle;
/*! \brief handle to tree builder class */
typedef void* TreeBuilderHandle;
/*! \brief handle to ensemble builder class */
typedef void* ModelBuilderHandle;
/*! \} */

TREELITE_DLL int TreeliteFreeModel(ModelHandle handle);

/*!
 * \defgroup model_builder
 * Model builder interface: build trees incrementally
 * \{
 */
/*!
 * \brief Create a new tree builder
 * \param out newly created tree builder
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteCreateTreeBuilder(TreeBuilderHandle* out);
/*!
 * \brief Delete a tree builder from memory
 * \param handle tree builder to remove
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteDeleteTreeBuilder(TreeBuilderHandle handle);
/*!
 * \brief Create an empty node within a tree
 * \param handle tree builder
 * \param node_key unique integer key to identify the new node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderCreateNode(TreeBuilderHandle handle,
                                               int node_key);
/*!
 * \brief Remove a node from a tree
 * \param handle tree builder
 * \param node_key unique integer key to identify the node to be removed
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderDeleteNode(TreeBuilderHandle handle,
                                               int node_key);
/*!
 * \brief Set a node as the root of a tree
 * \param handle tree builder
 * \param node_key unique integer key to identify the root node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetRootNode(TreeBuilderHandle handle,
                                                int node_key);
/*!
 * \brief Turn an empty node into a test node with numerical split.
 * The test is in the form [feature value] OP [threshold]. Depending on the
 * result of the test, either left or right child would be taken.
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param feature_id id of feature
 * \param opname binary operator to use in the test
 * \param threshold threshold value
 * \param default_left default direction for missing values
 * \param left_child_key unique integer key to identify the left child node
 * \param right_child_key unique integer key to identify the right child node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetNumericalTestNode(
                                             TreeBuilderHandle handle,
                                             int node_key, unsigned feature_id,
                                             const char* opname,
                                             float threshold, int default_left,
                                             int left_child_key,
                                             int right_child_key);
/*!
 * \brief Turn an empty node into a test node with categorical split.
 * A list defines all categories that would be classified as the left side.
 * Categories are integers ranging from 0 to (n-1), where n is the number of
 * categories in that particular feature. Let's assume n <= 64.
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param feature_id id of feature
 * \param left_categories list of categories belonging to the left child
 * \param left_categories_len length of left_cateogries
 * \param default_left default direction for missing values
 * \param left_child_key unique integer key to identify the left child node
 * \param right_child_key unique integer key to identify the right child node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetCategoricalTestNode(
                                          TreeBuilderHandle handle,
                                          int node_key, unsigned feature_id,
                                          const unsigned int* left_categories,
                                          size_t left_categories_len,
                                          int default_left,
                                          int left_child_key,
                                          int right_child_key);
/*!
 * \brief Turn an empty node into a leaf node
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param leaf_value leaf value (weight) of the leaf node
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetLeafNode(TreeBuilderHandle handle,
                                                int node_key,
                                                float leaf_value);
/*!
 * \brief Turn an empty node into a leaf vector node
 * The leaf vector (collection of multiple leaf weights per leaf node) is
 * useful for multi-class random forest classifier.
 * \param handle tree builder
 * \param node_key unique integer key to identify the node being modified;
 *                 this node needs to be empty
 * \param leaf_vector leaf vector of the leaf node
 * \param leaf_vector_len length of leaf_vector
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteTreeBuilderSetLeafVectorNode(TreeBuilderHandle handle,
                                                      int node_key,
                                                      const float* leaf_vector,
                                                      size_t leaf_vector_len);
/*!
 * \brief Create a new model builder
 * \param num_feature number of features used in model being built. We assume
 *                    that all feature indices are between 0 and
 *                    (num_feature - 1).
 * \param num_output_group number of output groups. Set to 1 for binary
 *                         classification and regression; >1 for multiclass
 *                         classification
 * \param random_forest_flag whether the model is a random forest. Set to 0 if
 *                           the model is gradient boosted trees. Any nonzero
 *                           value shall indicate that the model is a
 *                           random forest.
 * \param out newly created model builder
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteCreateModelBuilder(int num_feature,
                                            int num_output_group,
                                            int random_forest_flag,
                                            ModelBuilderHandle* out);
/*!
 * \brief Set a model parameter
 * \param handle model builder
 * \param name name of parameter
 * \param value value of parameter
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderSetModelParam(ModelBuilderHandle handle,
                                                   const char* name,
                                                   const char* value);
/*!
 * \brief Delete a model builder from memory
 * \param handle model builder to remove
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteDeleteModelBuilder(ModelBuilderHandle handle);
/*!
 * \brief Insert a tree at specified location
 * \param handle model builder
 * \param tree_builder builder for the tree to be inserted. The tree must not
 *                     be part of any other existing tree ensemble. Note:
 *                     The tree_builder argument will become unusuable after
 *                     the tree insertion. Should you want to modify the
 *                     tree afterwards, use GetTree(*) method to get a fresh
 *                     handle to the tree.
 * \param index index of the element before which to insert the tree;
 *              use -1 to insert at the end
 * \return index of the new tree within the ensemble; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderInsertTree(ModelBuilderHandle handle,
                                                TreeBuilderHandle tree_builder,
                                                int index);
/*!
 * \brief Get a reference to a tree in the ensemble
 * \param handle model builder
 * \param index index of the tree in the ensemble
 * \param out used to save reference to the tree
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderGetTree(ModelBuilderHandle handle,
                                             int index,
                                             TreeBuilderHandle *out);
/*!
 * \brief Remove a tree from the ensemble
 * \param handle model builder
 * \param index index of the tree that would be removed
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderDeleteTree(ModelBuilderHandle handle,
                                                int index);
/*!
 * \brief finalize the model and produce the in-memory representation
 * \param handle model builder
 * \param out used to save handle to in-memory representation of the finished
 *            model
 * \return 0 for success; -1 for failure
 */
TREELITE_DLL int TreeliteModelBuilderCommitModel(ModelBuilderHandle handle,
                                                 ModelHandle* out);
/*! \} */

#endif  /* TREELITE_C_API_H_ */
