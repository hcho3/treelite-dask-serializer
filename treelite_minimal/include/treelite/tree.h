/*!
 * Copyright 2020 by Contributors
 * \file tree.h
 * \brief model structure for tree ensemble
 * \author Hyunsu Cho
 */
#ifndef TREELITE_TREE_H_
#define TREELITE_TREE_H_

#include <treelite/base.h>
#include <dmlc/parameter.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <type_traits>
#include <limits>

namespace treelite {

struct PyBufferInterface1D {
  void* buf;
  char* format;
  size_t itemsize;
  size_t nitem;
};

struct PyBufferInterfaceTreeliteModel {
  std::vector<PyBufferInterface1D> frames;
  size_t ntree;
  size_t nframe_per_tree;
};

inline PyBufferInterface1D GetPyBufferFromVector(void* data, const char* format,
                                       size_t itemsize, size_t nitem) {
  return PyBufferInterface1D{data, const_cast<char*>(format), itemsize, nitem};
}

template <typename T>
inline PyBufferInterface1D GetPyBufferFromVector(std::vector<T>& vec) {
  static_assert(sizeof(uint8_t) == 1, "Assumed sizeof(uint8_t) == 1");
  if (!std::is_arithmetic<T>::value) {
    return GetPyBufferFromVector(static_cast<void*>(vec.data()), "=B",
                                 sizeof(uint8_t), vec.size() * sizeof(T));
  }
  const char* format = nullptr;
  switch (sizeof(T)) {
   case 1:
     format = (std::is_unsigned<T>::value ? "=B" : "=b");
     break;
   case 2:
     format = (std::is_unsigned<T>::value ? "=H" : "=h");
   case 4:
     if (std::is_integral<T>::value) {
       format = (std::is_unsigned<T>::value ? "=L" : "=l");
     } else {
       CHECK(std::is_floating_point<T>::value);
       format = "=f";
     }
     break;
   case 8:
     if (std::is_integral<T>::value) {
       format = (std::is_unsigned<T>::value ? "=Q" : "=q");
     } else {
       CHECK(std::is_floating_point<T>::value);
       format = "=d";
     }
     break;
   default:
     LOG(FATAL) << "Type not supported";
  }
  return GetPyBufferFromVector(static_cast<void*>(vec.data()), format, sizeof(T), vec.size());
}

/*! \brief in-memory representation of a decision tree */
class Tree {
 public:
  /*! \brief tree node */
  struct Node {
    void Init() {
      sindex_ = 0;
      missing_category_to_zero_ = false;
      data_count_present_ = sum_hess_present_ = gain_present_ = false;
    }
    /*! \brief store either leaf value or decision threshold */
    union Info {
      tl_float leaf_value;  // for leaf nodes
      tl_float threshold;   // for non-leaf nodes
    };
    /*! \brief pointer to left and right children */
    int32_t cleft_, cright_;
    /*! \brief feature split type */
    SplitFeatureType split_type_;
    /*!
     * \brief feature index used for the split
     * highest bit indicates default direction for missing values
     */
    uint32_t sindex_;
    /*! \brief storage for leaf value or decision threshold */
    Info info_;
    /*!
     * \brief operator to use for expression of form [fval] OP [threshold].
     * If the expression evaluates to true, take the left child;
     * otherwise, take the right child.
     */
    Operator cmp_;
    /* \brief Whether to convert missing value to zero.
     * Only applicable when split_type_ is set to kCategorical.
     * When this flag is set, it overrides the behavior of default_left().
     */
    bool missing_category_to_zero_;
    /*!
     * \brief number of data points whose traversal paths include this node.
     *        LightGBM models natively store this statistics.
     */
    bool data_count_present_;
    uint64_t data_count_;
    /*!
     * \brief sum of hessian values for all data points whose traversal paths
     *        include this node. This value is generally correlated positively
     *        with the data count. XGBoost models natively store this
     *        statistics.
     */
    bool sum_hess_present_;
    double sum_hess_;
    /*!
     * \brief change in loss that is attributed to a particular split
     */
    bool gain_present_;
    double gain_;
  };

  static_assert(std::is_pod<Node>::value, "Node must be a POD type");

 public:
  inline std::vector<PyBufferInterface1D> GetPyBuffer() {
    return {GetPyBufferFromVector(nodes_), GetPyBufferFromVector(leaf_vector_),
      GetPyBufferFromVector(leaf_vector_offset_), GetPyBufferFromVector(left_categories_),
      GetPyBufferFromVector(left_categories_offset_)};
  }

 private:
  // vector of nodes
  std::vector<Node> nodes_;
  std::vector<tl_float> leaf_vector_;
  std::vector<size_t> leaf_vector_offset_;
  std::vector<uint32_t> left_categories_;
  std::vector<size_t> left_categories_offset_;

  // allocate a new node
  inline int AllocNode() {
    int nd = num_nodes++;
    CHECK_LT(num_nodes, std::numeric_limits<int>::max())
        << "number of nodes in the tree exceed 2^31";
    CHECK_EQ(nodes_.size(), static_cast<size_t>(nd));
    for (int nid = nd; nid < num_nodes; ++nid) {
      leaf_vector_offset_.push_back(leaf_vector_offset_.back());
      left_categories_offset_.push_back(left_categories_offset_.back());
      nodes_.emplace_back();
      nodes_.back().Init();
    }
    return nd;
  }

 public:
  /*! \brief number of nodes */
  int num_nodes;
  /*! \brief initialize the model with a single root node */
  inline void Init() {
    num_nodes = 1;
    leaf_vector_.clear();
    leaf_vector_offset_ = {0, 0};
    left_categories_.clear();
    left_categories_offset_ = {0, 0};
    nodes_.clear();
    nodes_.emplace_back();
    nodes_[0].Init();
    SetLeaf(0, 0.0f);
  }
  /*!
   * \brief add child nodes to node
   * \param nid node id to add children to
   */
  inline void AddChilds(int nid) {
    const int cleft = this->AllocNode();
    const int cright = this->AllocNode();
    nodes_[nid].cleft_ = cleft;
    nodes_[nid].cright_ = cright;
  }

  /*!
   * \brief get list of all categorical features that have appeared anywhere in tree
   * \return list of all categorical features used
   */
  inline std::vector<unsigned> GetCategoricalFeatures() const;

  /** Getters **/
  /*!
   * \brief index of the node's left child
   * \param nid ID of node being queried
   */
  inline int LeftChild(int nid) const;
  /*!
   * \brief index of the node's right child
   * \param nid ID of node being queried
   */
  inline int RightChild(int nid) const;
  /*!
   * \brief index of the node's "default" child, used when feature is missing
   * \param nid ID of node being queried
   */
  inline int DefaultChild(int nid) const;
  /*!
   * \brief feature index of the node's split condition
   * \param nid ID of node being queried
   */
  inline uint32_t SplitIndex(int nid) const;
  /*!
   * \brief whether to use the left child node, when the feature in the split condition is missing
   * \param nid ID of node being queried
   */
  inline bool DefaultLeft(int nid) const;
  /*!
   * \brief whether the node is leaf node
   * \param nid ID of node being queried
   */
  inline bool IsLeaf(int nid) const;
  /*!
   * \brief get leaf value of the leaf node 
   * \param nid ID of node being queried
   */
  inline tl_float LeafValue(int nid) const;
  /*!
   * \brief get leaf vector of the leaf node; useful for multi-class random forest classifier 
   * \param nid ID of node being queried
   */
  inline std::vector<tl_float> LeafVector(int nid) const;
  /*!
   * \brief tests whether the leaf node has a non-empty leaf vector 
   * \param nid ID of node being queried
   */
  inline bool HasLeafVector(int nid) const;
  /*!
   * \brief get threshold of the node 
   * \param nid ID of node being queried
   */
  inline tl_float Threshold(int nid) const;
  /*!
   * \brief get comparison operator 
   * \param nid ID of node being queried
   */
  inline Operator ComparisonOp(int nid) const;
  /*!
   * \brief Get list of all categories belonging to the left child node. Categories not in this
   *        list will belong to the right child node. Categories are integers ranging from 0 to
   *        (n-1), where n is the number of categories in that particular feature. This list is
   *        assumed to be in ascending order.
   * \param nid ID of node being queried
   */
  inline std::vector<uint32_t> LeftCategories(int nid) const;
  /*!
   * \brief get feature split type 
   * \param nid ID of node being queried
   */
  inline SplitFeatureType SplitType(int nid) const;
  /*!
   * \brief test whether this node has data count 
   * \param nid ID of node being queried
   */
  inline bool HasDataCount(int nid) const;
  /*!
   * \brief get data count 
   * \param nid ID of node being queried
   */
  inline uint64_t DataCount(int nid) const;
  /*!
   * \brief test whether this node has hessian sum 
   * \param nid ID of node being queried
   */
  inline bool HasSumHess(int nid) const;
  /*!
   * \brief get hessian sum 
   * \param nid ID of node being queried
   */
  inline double SumHess(int nid) const;
  /*!
   * \brief test whether this node has gain value 
   * \param nid ID of node being queried
   */
  inline bool HasGain(int nid) const;
  /*!
   * \brief get gain value 
   * \param nid ID of node being queried
   */
  inline double Gain(int nid) const;
  /*!
   * \brief test whether missing values should be converted into zero; only applicable for
   *        categorical splits
   * \param nid ID of node being queried
   */
  inline bool MissingCategoryToZero(int nid) const;

  /** Setters **/
  /*!
   * \brief create a numerical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param threshold threshold value
   * \param default_left the default direction when feature is unknown
   * \param cmp comparison operator to compare between feature value and
   *            threshold
   */
  inline void SetNumericalSplit(int nid, unsigned split_index, tl_float threshold,
      bool default_left, Operator cmp);
  /*!
   * \brief create a categorical split
   * \param nid ID of node being updated
   * \param split_index feature index to split
   * \param threshold threshold value
   * \param default_left the default direction when feature is unknown
   * \param cmp comparison operator to compare between feature value and
   *            threshold
   */
  inline void SetCategoricalSplit(int nid, unsigned split_index, bool default_left,
      bool missing_category_to_zero, const std::vector<uint32_t>& left_categories);
  /*!
   * \brief set the leaf value of the node
   * \param nid ID of node being updated
   * \param value leaf value
   */
  inline void SetLeaf(int nid, tl_float value);
  /*!
   * \brief set the leaf vector of the node; useful for multi-class random forest classifier
   * \param nid ID of node being updated
   * \param leaf_vector leaf vector
   */
  inline void SetLeafVector(int nid, const std::vector<tl_float>& leaf_vector);
  /*!
   * \brief set the hessian sum of the node
   * \param nid ID of node being updated
   * \param sum_hess hessian sum
   */
  inline void SetSumHess(int nid, double sum_hess);
  /*!
   * \brief set the data count of the node
   * \param nid ID of node being updated
   * \param data_count data count
   */
  inline void SetDataCount(int nid, uint64_t data_count);
  /*!
   * \brief set the gain value of the node
   * \param nid ID of node being updated
   * \param gain gain value
   */
  inline void SetGain(int nid, double gain);

  void Serialize(dmlc::Stream* fo) const;
  void Deserialize(dmlc::Stream* fi);
};

struct ModelParam : public dmlc::Parameter<ModelParam> {
  /*!
  * \defgroup model_param
  * Extra parameters for tree ensemble models
  * \{
  */
  /*!
   * \brief name of prediction transform function
   *
   * This parameter specifies how to transform raw margin values into
   * final predictions. By default, this is set to `'identity'`, which
   * means no transformation.
   *
   * For the **multi-class classification task**, `pred_transfrom` must be one
   * of the following values:
   * \snippet src/compiler/pred_transform.cc pred_transform_multiclass_db
   *
   * For **all other tasks** (e.g. regression, binary classification, ranking
   * etc.), `pred_transfrom` must be one of the following values:
   * \snippet src/compiler/pred_transform.cc pred_transform_db
   *
   */
  std::string pred_transform;
  /*!
   * \brief scaling parameter for sigmoid function
   * `sigmoid(x) = 1 / (1 + exp(-alpha * x))`
   *
   * This parameter is used only when `pred_transform` is set to `'sigmoid'`.
   * It must be strictly positive; if unspecified, it is set to 1.0.
   */
  float sigmoid_alpha;
  /*!
   * \brief global bias of the model
   *
   * Predicted margin scores of all instances will be adjusted by the global
   * bias. If unspecified, the bias is set to zero.
   */
  float global_bias;
  /*! \} */

  // declare parameters
  DMLC_DECLARE_PARAMETER(ModelParam) {
    DMLC_DECLARE_FIELD(pred_transform).set_default("identity")
      .describe("name of prediction transform function");
    DMLC_DECLARE_FIELD(sigmoid_alpha).set_default(1.0f)
      .set_lower_bound(0.0f)
      .describe("scaling parameter for sigmoid function");
    DMLC_DECLARE_FIELD(global_bias).set_default(0.0f)
      .describe("global bias of the model");
  }
};

inline void InitParamAndCheck(ModelParam* param,
                  const std::vector<std::pair<std::string, std::string>> cfg) {
  auto unknown = param->InitAllowUnknown(cfg);
  if (unknown.size() > 0) {
    std::ostringstream oss;
    for (const auto& kv : unknown) {
      oss << kv.first << ", ";
    }
    LOG(INFO) << "\033[1;31mWarning: Unknown parameters found; "
              << "they have been ignored\u001B[0m: " << oss.str();
  }
}

/*! \brief thin wrapper for tree ensemble model */
struct Model {
  /*! \brief member trees */
  std::vector<Tree> trees;
  /*!
   * \brief number of features used for the model.
   * It is assumed that all feature indices are between 0 and [num_feature]-1.
   */
  int num_feature;
  /*! \brief number of output groups -- for multi-class classification
   *  Set to 1 for everything else */
  int num_output_group;
  /*! \brief flag for random forest;
   *  True for random forests and False for gradient boosted trees */
  bool random_forest_flag;
  /*! \brief extra parameters */
  ModelParam param;

  /*! \brief disable copy; use default move */
  Model() {
    param.Init(std::vector<std::pair<std::string, std::string>>());
  }
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

  void Serialize(dmlc::Stream* fo) const;
  void Deserialize(dmlc::Stream* fi);

  inline PyBufferInterfaceTreeliteModel GetPyBuffer() {
    PyBufferInterfaceTreeliteModel buffer;
    buffer.nframe_per_tree = 0;
    for (auto& tree : trees) {
      std::vector<PyBufferInterface1D> frames = tree.GetPyBuffer();
      buffer.frames.insert(buffer.frames.end(), frames.begin(), frames.end());
      if (buffer.nframe_per_tree > 0) {
        CHECK_EQ(buffer.nframe_per_tree, frames.size());
      } else {
        buffer.nframe_per_tree = frames.size();
      }
    }
    buffer.ntree = trees.size();
    return buffer;
  }
};

/** Implementations **/
inline std::vector<unsigned>
Tree::GetCategoricalFeatures() const {
  std::unordered_map<unsigned, bool> tmp;
  for (int nid = 0; nid < num_nodes; ++nid) {
    const SplitFeatureType type = SplitType(nid);
    if (type != SplitFeatureType::kNone) {
      const bool flag = (type == SplitFeatureType::kCategorical);
      const uint32_t split_index = SplitIndex(nid);
      if (tmp.count(split_index) == 0) {
        tmp[split_index] = flag;
      } else {
        CHECK_EQ(tmp[split_index], flag) << "Feature " << split_index
          << " cannot be simultaneously be categorical and numerical.";
      }
    }
  }
  std::vector<unsigned> result;
  for (const auto& kv : tmp) {
    if (kv.second) {
      result.push_back(kv.first);
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

inline int
Tree::LeftChild(int nid) const {
  return nodes_[nid].cleft_;
}

inline int
Tree::RightChild(int nid) const {
  return nodes_[nid].cright_;
}

inline int
Tree::DefaultChild(int nid) const {
  return DefaultLeft(nid) ? LeftChild(nid) : RightChild(nid);
}

inline uint32_t
Tree::SplitIndex(int nid) const {
  return (nodes_[nid].sindex_ & ((1U << 31) - 1U));
}

inline bool
Tree::DefaultLeft(int nid) const {
  return (nodes_[nid].sindex_ >> 31) != 0;
}

inline bool
Tree::IsLeaf(int nid) const {
  return nodes_[nid].cleft_ == -1;
}

inline tl_float
Tree::LeafValue(int nid) const {
  return (nodes_[nid].info_).leaf_value;
}

inline std::vector<tl_float>
Tree::LeafVector(int nid) const {
  CHECK_LE(nid, leaf_vector_offset_.size());
  return std::vector<tl_float>(&leaf_vector_[leaf_vector_offset_[nid]],
                               &leaf_vector_[leaf_vector_offset_[nid + 1]]);
}

inline bool
Tree::HasLeafVector(int nid) const {
  CHECK_LE(nid, leaf_vector_offset_.size());
  return leaf_vector_offset_[nid] != leaf_vector_offset_[nid + 1];
}

inline tl_float
Tree::Threshold(int nid) const {
  return (nodes_[nid].info_).threshold;
}

inline Operator
Tree::ComparisonOp(int nid) const {
  return nodes_[nid].cmp_;
}

inline std::vector<uint32_t>
Tree::LeftCategories(int nid) const {
  CHECK_LE(nid, left_categories_offset_.size());
  return std::vector<uint32_t>(&left_categories_[left_categories_offset_[nid]],
                               &left_categories_[left_categories_offset_[nid + 1]]);
}

inline SplitFeatureType
Tree::SplitType(int nid) const {
  return nodes_[nid].split_type_;
}

inline bool
Tree::HasDataCount(int nid) const {
  return nodes_[nid].data_count_present_;
}

inline uint64_t
Tree::DataCount(int nid) const {
  return nodes_[nid].data_count_;
}

inline bool
Tree::HasSumHess(int nid) const {
  return nodes_[nid].sum_hess_present_;
}

inline double
Tree::SumHess(int nid) const {
  return nodes_[nid].sum_hess_;
}

inline bool
Tree::HasGain(int nid) const {
  return nodes_[nid].gain_present_;
}

inline double
Tree::Gain(int nid) const {
  return nodes_[nid].gain_;
}

inline bool
Tree::MissingCategoryToZero(int nid) const {
  return nodes_[nid].missing_category_to_zero_;
}

inline void
Tree::SetNumericalSplit(int nid, unsigned split_index, tl_float threshold,
    bool default_left, Operator cmp) {
  Node& node = nodes_[nid];
  CHECK_LT(split_index, (1U << 31) - 1) << "split_index too big";
  if (default_left) split_index |= (1U << 31);
  node.sindex_ = split_index;
  (node.info_).threshold = threshold;
  node.cmp_ = cmp;
  node.split_type_ = SplitFeatureType::kNumerical;
}

inline void
Tree::SetCategoricalSplit(int nid, unsigned split_index, bool default_left,
    bool missing_category_to_zero, const std::vector<uint32_t>& node_left_categories) {
  CHECK_LT(split_index, (1U << 31) - 1) << "split_index too big";

  const size_t end_oft = left_categories_offset_.back();
  const size_t new_end_oft = end_oft + node_left_categories.size();
  CHECK_EQ(end_oft, left_categories_.size());
  CHECK(std::all_of(left_categories_offset_.begin() + (nid + 1), left_categories_offset_.end(),
                    [end_oft](size_t x) { return (x == end_oft); }));
    // Hopefully we won't have to move any element as we add node_left_categories for node nid
  left_categories_.insert(left_categories_.end(),
                          node_left_categories.begin(), node_left_categories.end());
  CHECK_EQ(new_end_oft, left_categories_.size());
  std::for_each(left_categories_offset_.begin() + (nid + 1), left_categories_offset_.end(),
                [new_end_oft](size_t& x) { x = new_end_oft; });
  std::sort(left_categories_.begin() + end_oft, left_categories_.end());

  Node& node = nodes_[nid];
  if (default_left) split_index |= (1U << 31);
  node.sindex_ = split_index;
  node.split_type_ = SplitFeatureType::kCategorical;
  node.missing_category_to_zero_ = missing_category_to_zero;
}

inline void
Tree::SetLeaf(int nid, tl_float value) {
  Node& node = nodes_[nid];
  (node.info_).leaf_value = value;
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

inline void
Tree::SetLeafVector(int nid, const std::vector<tl_float>& node_leaf_vector) {
  const size_t end_oft = leaf_vector_offset_.back();
  const size_t new_end_oft = end_oft + node_leaf_vector.size();
  CHECK_EQ(end_oft, leaf_vector_.size());
  CHECK(std::all_of(leaf_vector_offset_.begin() + (nid + 1), leaf_vector_offset_.end(),
                    [end_oft](size_t x) { return (x == end_oft); }));
    // Hopefully we won't have to move any element as we add leaf vector elements for node nid
  leaf_vector_.insert(leaf_vector_.end(), node_leaf_vector.begin(), node_leaf_vector.end());
  CHECK_EQ(new_end_oft, leaf_vector_.size());
  std::for_each(leaf_vector_offset_.begin() + (nid + 1), leaf_vector_offset_.end(),
                [new_end_oft](size_t& x) { x = new_end_oft; });

  Node& node = nodes_[nid];
  node.cleft_ = -1;
  node.cright_ = -1;
  node.split_type_ = SplitFeatureType::kNone;
}

inline void
Tree::SetSumHess(int nid, double sum_hess) {
  Node& node = nodes_[nid];
  node.sum_hess_ = sum_hess;
  node.sum_hess_present_ = true;
}

inline void
Tree::SetDataCount(int nid, uint64_t data_count) {
  Node& node = nodes_[nid];
  node.data_count_ = data_count;
  node.data_count_present_ = true;
}

inline void
Tree::SetGain(int nid, double gain) {
  Node& node = nodes_[nid];
  node.gain_ = gain;
  node.gain_present_ = true;
}

}  // namespace treelite
#endif  // TREELITE_TREE_H_
