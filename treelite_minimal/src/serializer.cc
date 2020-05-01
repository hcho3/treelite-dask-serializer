/*!
 * Copyright 2020 by Contributors
 * \file serializer.cc
 * \brief Serialization of Tree and Model objects
 * \author Hyunsu Cho
 */

#include <treelite/tree.h>
#include <dmlc/serializer.h>
#include <dmlc/io.h>
#include <dmlc/timer.h>

namespace treelite {

void Tree::Serialize(dmlc::Stream* fo) const {
  fo->Write(num_nodes);
  fo->Write(leaf_vector_);
  fo->Write(leaf_vector_offset_);
  fo->Write(left_categories_);
  fo->Write(left_categories_offset_);
  uint64_t sz = static_cast<uint64_t>(nodes_.size());
  fo->Write(sz);
  fo->Write(nodes_.data(), sz * sizeof(Tree::Node));

  // Sanity check
  CHECK_EQ(nodes_.size(), num_nodes);
  CHECK_EQ(nodes_.size() + 1, leaf_vector_offset_.size());
  CHECK_EQ(leaf_vector_offset_.back(), leaf_vector_.size());
  CHECK_EQ(nodes_.size() + 1, left_categories_offset_.size());
  CHECK_EQ(left_categories_offset_.back(), left_categories_.size());
}

void Tree::Deserialize(dmlc::Stream* fi) {
  fi->Read(&num_nodes);
  fi->Read(&leaf_vector_);
  fi->Read(&leaf_vector_offset_);
  fi->Read(&left_categories_);
  fi->Read(&left_categories_offset_);
  uint64_t sz = 0;
  fi->Read(&sz);
  nodes_.clear();
  nodes_.resize(sz, Node(this, -1));
  fi->Read(nodes_.data(), sz * sizeof(Node));
  for (uint64_t i = 0; i < sz; ++i) {
    nodes_[i].nid_ = i;
  }

  // Sanity check
  CHECK_EQ(nodes_.size(), num_nodes);
  CHECK_EQ(nodes_.size() + 1, leaf_vector_offset_.size());
  CHECK_EQ(leaf_vector_offset_.back(), leaf_vector_.size());
  CHECK_EQ(nodes_.size() + 1, left_categories_offset_.size());
  CHECK_EQ(left_categories_offset_.back(), left_categories_.size());
}

void Model::Serialize(dmlc::Stream* fo) const {
  fo->Write(num_feature);
  fo->Write(num_output_group);
  fo->Write(random_forest_flag);
  fo->Write(&param, sizeof(param));
  uint64_t sz = static_cast<uint64_t>(trees.size());
  fo->Write(sz);
  for (const Tree& tree : trees) {
    tree.Serialize(fo);
  }
}

void Model::Deserialize(dmlc::Stream* fi) {
  fi->Read(&num_feature);
  fi->Read(&num_output_group);
  fi->Read(&random_forest_flag);
  fi->Read(&param, sizeof(param));
  uint64_t sz = 0;
  fi->Read(&sz);
  for (uint64_t i = 0; i < sz; ++i) {
    Tree tree;
    tree.Deserialize(fi);
    trees.push_back(std::move(tree));
  }
}

}  // namespace treelite
