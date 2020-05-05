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

namespace dmlc {
namespace serializer {

template <typename T>
struct Handler<treelite::ContiguousArray<T>> {
  inline static void Write(Stream* strm, const treelite::ContiguousArray<T>& data) {
    uint64_t sz = static_cast<uint64_t>(data.Size());
    strm->Write(sz);
    strm->Write(data.Data(), sz * sizeof(T));
  }

  inline static bool Read(Stream* strm, treelite::ContiguousArray<T>* data) {
    uint64_t sz;
    bool status = strm->Read(&sz);
    if (!status) {
      return false;
    }
    data->Resize(sz);
    return strm->Read(data->Data(), sz * sizeof(T));
  }
};

}  // namespace serializer
}  // namespace dmlc

namespace treelite {

void Tree::Serialize(dmlc::Stream* fo) const {
  fo->Write(num_nodes);
  fo->Write(leaf_vector_);
  fo->Write(leaf_vector_offset_);
  fo->Write(left_categories_);
  fo->Write(left_categories_offset_);
  uint64_t sz = static_cast<uint64_t>(nodes_.Size());
  fo->Write(sz);
  fo->Write(nodes_.Data(), sz * sizeof(Tree::Node));

  // Sanity check
  CHECK_EQ(nodes_.Size(), num_nodes);
  CHECK_EQ(nodes_.Size() + 1, leaf_vector_offset_.Size());
  CHECK_EQ(leaf_vector_offset_.Back(), leaf_vector_.Size());
  CHECK_EQ(nodes_.Size() + 1, left_categories_offset_.Size());
  CHECK_EQ(left_categories_offset_.Back(), left_categories_.Size());
}

void Tree::Deserialize(dmlc::Stream* fi) {
  fi->Read(&num_nodes);
  fi->Read(&leaf_vector_);
  fi->Read(&leaf_vector_offset_);
  fi->Read(&left_categories_);
  fi->Read(&left_categories_offset_);
  uint64_t sz = 0;
  fi->Read(&sz);
  nodes_.Resize(sz);
  fi->Read(nodes_.Data(), sz * sizeof(Node));

  // Sanity check
  CHECK_EQ(nodes_.Size(), num_nodes);
  CHECK_EQ(nodes_.Size() + 1, leaf_vector_offset_.Size());
  CHECK_EQ(leaf_vector_offset_.Back(), leaf_vector_.Size());
  CHECK_EQ(nodes_.Size() + 1, left_categories_offset_.Size());
  CHECK_EQ(left_categories_offset_.Back(), left_categories_.Size());
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
