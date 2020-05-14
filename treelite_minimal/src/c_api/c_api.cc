/*!
 * Copyright (c) 2020 by Contributors
 * \file c_api.cc
 * \author Hyunsu Cho
 * \brief C API of treelite, used for interfacing with other languages
 */

#include <treelite/c_api.h>
#include <treelite/tree.h>
#include <treelite/frontend.h>

#include "./c_api_error.h"

using namespace treelite;

int TreeliteFreeModel(ModelHandle handle) {
  API_BEGIN();
  delete static_cast<Model*>(handle);
  API_END();
}

int TreeliteCreateTreeBuilder(TreeBuilderHandle* out) {
  API_BEGIN();
  auto builder = new frontend::TreeBuilder();
  *out = static_cast<TreeBuilderHandle>(builder);
  API_END();
}

int TreeliteDeleteTreeBuilder(TreeBuilderHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::TreeBuilder*>(handle);
  API_END();
}

int TreeliteTreeBuilderCreateNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->CreateNode(node_key);
  API_END();
}

int TreeliteTreeBuilderDeleteNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->DeleteNode(node_key);
  API_END();
}

int TreeliteTreeBuilderSetRootNode(TreeBuilderHandle handle, int node_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetRootNode(node_key);
  API_END();
}

int TreeliteTreeBuilderSetNumericalTestNode(TreeBuilderHandle handle,
                                            int node_key, unsigned feature_id,
                                            const char* opname,
                                            float threshold, int default_left,
                                            int left_child_key,
                                            int right_child_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetNumericalTestNode(node_key, feature_id, opname, static_cast<tl_float>(threshold),
                                (default_left != 0), left_child_key, right_child_key);
  API_END();
}

int TreeliteTreeBuilderSetCategoricalTestNode(TreeBuilderHandle handle,
                                              int node_key, unsigned feature_id,
                                              const unsigned int* left_categories,
                                              size_t left_categories_len,
                                              int default_left,
                                              int left_child_key,
                                              int right_child_key) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<uint32_t> vec(left_categories_len);
  for (size_t i = 0; i < left_categories_len; ++i) {
    CHECK(left_categories[i] <= std::numeric_limits<uint32_t>::max());
    vec[i] = static_cast<uint32_t>(left_categories[i]);
  }
  builder->SetCategoricalTestNode(node_key, feature_id, vec, (default_left != 0),
                                  left_child_key, right_child_key);
  API_END();
}

int TreeliteTreeBuilderSetLeafNode(TreeBuilderHandle handle, int node_key, float leaf_value) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  builder->SetLeafNode(node_key, static_cast<tl_float>(leaf_value));
  API_END();
}

int TreeliteTreeBuilderSetLeafVectorNode(TreeBuilderHandle handle,
                                         int node_key,
                                         const float* leaf_vector,
                                         size_t leaf_vector_len) {
  API_BEGIN();
  auto builder = static_cast<frontend::TreeBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted TreeBuilder object";
  std::vector<tl_float> vec(leaf_vector_len);
  for (size_t i = 0; i < leaf_vector_len; ++i) {
    vec[i] = static_cast<tl_float>(leaf_vector[i]);
  }
  builder->SetLeafVectorNode(node_key, vec);
  API_END();
}

int TreeliteCreateModelBuilder(int num_feature,
                               int num_output_group,
                               int random_forest_flag,
                               ModelBuilderHandle* out) {
  API_BEGIN();
  auto builder = new frontend::ModelBuilder(num_feature, num_output_group,
                                            (random_forest_flag != 0));
  *out = static_cast<ModelBuilderHandle>(builder);
  API_END();
}

int TreeliteModelBuilderSetModelParam(ModelBuilderHandle handle,
                                      const char* name,
                                      const char* value) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  builder->SetModelParam(name, value);
  API_END();
}

int TreeliteDeleteModelBuilder(ModelBuilderHandle handle) {
  API_BEGIN();
  delete static_cast<frontend::ModelBuilder*>(handle);
  API_END();
}

int TreeliteModelBuilderInsertTree(ModelBuilderHandle handle,
                                   TreeBuilderHandle tree_builder_handle,
                                   int index) {
  API_BEGIN();
  auto model_builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(model_builder) << "Detected dangling reference to deleted ModelBuilder object";
  auto tree_builder = static_cast<frontend::TreeBuilder*>(tree_builder_handle);
  CHECK(tree_builder) << "Detected dangling reference to deleted TreeBuilder object";
  return model_builder->InsertTree(tree_builder, index);
  API_END();
}

int TreeliteModelBuilderGetTree(ModelBuilderHandle handle, int index,
                                TreeBuilderHandle *out) {
  API_BEGIN();
  auto model_builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(model_builder) << "Detected dangling reference to deleted ModelBuilder object";
  auto tree_builder = model_builder->GetTree(index);
  CHECK(tree_builder) << "Detected dangling reference to deleted TreeBuilder object";
  *out = static_cast<TreeBuilderHandle>(tree_builder);
  API_END();
}

int TreeliteModelBuilderDeleteTree(ModelBuilderHandle handle, int index) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  builder->DeleteTree(index);
  API_END();
}

int TreeliteModelBuilderCommitModel(ModelBuilderHandle handle,
                                    ModelHandle* out) {
  API_BEGIN();
  auto builder = static_cast<frontend::ModelBuilder*>(handle);
  CHECK(builder) << "Detected dangling reference to deleted ModelBuilder object";
  Model* model = new Model();
  builder->CommitModel(model);
  *out = static_cast<ModelHandle>(model);
  API_END();
}
