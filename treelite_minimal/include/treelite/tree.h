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

#define __TREELITE_STR(x) #x
#define _TREELITE_STR(x) __TREELITE_STR(x)

#define TREELITE_MAX_PRED_TRANSFORM_LENGTH 256

namespace treelite {

struct PyBufferFrame {
  void* buf;
  char* format;
  size_t itemsize;
  size_t nitem;
};

template <typename T>
class ContiguousArray {
 public:
  ContiguousArray() : buffer_(nullptr), size_(0), capacity_(0), owned_buffer_(true) {}
  ~ContiguousArray() {
    if (buffer_ && owned_buffer_) {
      std::free(buffer_);
    }
  }
  // NOTE: use Clone to make deep copy; copy constructors disabled
  ContiguousArray(const ContiguousArray&) = delete;
  ContiguousArray& operator=(const ContiguousArray&) = delete;
  ContiguousArray(ContiguousArray&& other)
      : buffer_(other.buffer_), size_(other.size_), capacity_(other.capacity_),
        owned_buffer_(other.owned_buffer_) {
    other.buffer_ = nullptr;
    other.size_ = other.capacity_ = 0;
  }
  ContiguousArray& operator=(ContiguousArray&& other) {
    if (buffer_ && owned_buffer_) {
      std::free(buffer_);
    }
    buffer_ = other.buffer_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    owned_buffer_ = other.owned_buffer_;
    other.buffer_ = nullptr;
    other.size_ = other.capacity_ = 0;
  }
  inline ContiguousArray Clone() const {
    ContiguousArray clone;
    clone.buffer_ = static_cast<T*>(std::malloc(sizeof(T) * capacity_));
    CHECK(clone.buffer_);
    std::memcpy(clone.buffer_, buffer_, sizeof(T) * size_);
    clone.size_ = size_;
    clone.capacity_ = capacity_;
    clone.owned_buffer_ = true;
    return clone;
  }
  inline void UseForeignBuffer(void* prealloc_buf, size_t size) {
    if (buffer_ && owned_buffer_) {
      std::free(buffer_);
    }
    buffer_ = static_cast<T*>(prealloc_buf);
    size_ = size;
    capacity_ = size;
    owned_buffer_ = false;
  }
  inline T* Data() {
    return buffer_;
  }
  inline const T* Data() const {
    return buffer_;
  }
  inline T* End() { 
    return &buffer_[Size()];
  }
  inline const T* End() const {
    return &buffer_[Size()];
  }
  inline T& Back() {
    return buffer_[Size() - 1];
  }
  inline const T& Back() const {
    return buffer_[Size() - 1];
  }
  inline size_t Size() const {
    return size_;
  }
  inline void Reserve(size_t newsize) {
    CHECK(owned_buffer_) << "Cannot resize when using a foreign buffer; clone first";
    T* newbuf = static_cast<T*>(std::realloc(static_cast<void*>(buffer_), sizeof(T) * newsize));
    CHECK(newbuf);
    buffer_ = newbuf;
    capacity_ = newsize;
  }
  inline void Resize(size_t newsize) {
    CHECK(owned_buffer_) << "Cannot resize when using a foreign buffer; clone first";
    if (newsize > capacity_) {
      size_t newcapacity = capacity_;
      if (newcapacity == 0) {
        newcapacity = 1;
      }
      while (newcapacity <= newsize) {
        newcapacity *= 2;
      }
      Reserve(newcapacity);
    }
    size_ = newsize;
  }
  inline void Resize(size_t newsize, T t) {
    CHECK(owned_buffer_) << "Cannot resize when using a foreign buffer; clone first";
    size_t oldsize = Size();
    Resize(newsize);
    for (size_t i = oldsize; i < newsize; ++i) {
      buffer_[i] = t;
    }
  }
  inline void Clear() {
    CHECK(owned_buffer_) << "Cannot clear when using a foreign buffer; clone first";
    Resize(0);
  }
  inline void PushBack(T t) {
    CHECK(owned_buffer_) << "Cannot add element when using a foreign buffer; clone first";
    if (size_ == capacity_) {
      Reserve(capacity_ * 2);
    }
    buffer_[size_++] = t;
  }
  inline void Extend(const std::vector<T>& other) {
    CHECK(owned_buffer_) << "Cannot add elements when using a foreign buffer; clone first";
    size_t newsize = size_ + other.size();
    if (newsize > capacity_) {
      size_t newcapacity = capacity_;
      if (newcapacity == 0) {
        newcapacity = 1;
      }
      while (newcapacity <= newsize) {
        newcapacity *= 2;
      }
      Reserve(newcapacity);
    }
    std::memcpy(&buffer_[size_], static_cast<const void*>(other.data()), sizeof(T) * other.size());
    size_ = newsize;
  }
  inline T& operator[](size_t idx) {
    return buffer_[idx];
  } 
  inline const T& operator[](size_t idx) const {
    return buffer_[idx];
  } 
  static_assert(std::is_pod<T>::value, "T must be POD");
 private:
  T* buffer_;
  size_t size_;
  size_t capacity_;
  bool owned_buffer_;
};

/*! \brief in-memory representation of a decision tree */
class Tree {
 public:
  /*! \brief tree node */
  struct Node {
    void Init() {
      cleft_ = cright_ = -1;
      sindex_ = 0;
      info_.leaf_value = 0.0f;
      info_.threshold = 0.0f;
      data_count_ = 0;
      sum_hess_ = gain_ = 0.0;
      missing_category_to_zero_ = false;
      data_count_present_ = sum_hess_present_ = gain_present_ = false;
      split_type_ = SplitFeatureType::kNone;
      cmp_ = Operator::kNone;
      pad_ = 0;
    }
    /*! \brief store either leaf value or decision threshold */
    union Info {
      tl_float leaf_value;  // for leaf nodes
      tl_float threshold;   // for non-leaf nodes
    };
    /*! \brief pointer to left and right children */
    int32_t cleft_, cright_;
    /*!
     * \brief feature index used for the split
     * highest bit indicates default direction for missing values
     */
    uint32_t sindex_;
    /*! \brief storage for leaf value or decision threshold */
    Info info_;
    /*!
     * \brief number of data points whose traversal paths include this node.
     *        LightGBM models natively store this statistics.
     */
    uint64_t data_count_;
    /*!
     * \brief sum of hessian values for all data points whose traversal paths
     *        include this node. This value is generally correlated positively
     *        with the data count. XGBoost models natively store this
     *        statistics.
     */
    double sum_hess_;
    /*!
     * \brief change in loss that is attributed to a particular split
     */
    double gain_;
    /*! \brief feature split type */
    SplitFeatureType split_type_;
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
    /*! \brief whether data_count_ field is present */
    bool data_count_present_;
    /*! \brief whether sum_hess_ field is present */
    bool sum_hess_present_;
    /*! \brief whether gain_present_ field is present */
    bool gain_present_;
    // padding
    uint16_t pad_;
  };

  static_assert(std::is_pod<Node>::value, "Node must be a POD type");
  static_assert(sizeof(Node) == 48, "Node must be 48 bytes");

  inline std::vector<PyBufferFrame> GetPyBuffer();
  inline void InitFromPyBuffer(std::vector<PyBufferFrame> frames);

 private:
  // vector of nodes
  ContiguousArray<Node> nodes_;
  ContiguousArray<tl_float> leaf_vector_;
  ContiguousArray<size_t> leaf_vector_offset_;
  ContiguousArray<uint32_t> left_categories_;
  ContiguousArray<size_t> left_categories_offset_;

  // allocate a new node
  inline int AllocNode() {
    int nd = num_nodes++;
    CHECK_LT(num_nodes, std::numeric_limits<int>::max())
        << "number of nodes in the tree exceed 2^31";
    CHECK_EQ(nodes_.Size(), static_cast<size_t>(nd));
    for (int nid = nd; nid < num_nodes; ++nid) {
      leaf_vector_offset_.PushBack(leaf_vector_offset_.Back());
      left_categories_offset_.PushBack(left_categories_offset_.Back());
      nodes_.Resize(nodes_.Size() + 1);
      nodes_.Back().Init();
    }
    return nd;
  }

 public:
  /*! \brief number of nodes */
  int num_nodes;
  /*! \brief initialize the model with a single root node */
  inline void Init() {
    num_nodes = 1;
    leaf_vector_.Clear();
    leaf_vector_offset_.Resize(2, 0);
    left_categories_.Clear();
    left_categories_offset_.Resize(2, 0);
    nodes_.Resize(1);
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

  void ReferenceSerialize(dmlc::Stream* fo) const;
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
  char pred_transform[TREELITE_MAX_PRED_TRANSFORM_LENGTH];
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

  ModelParam() {
    std::memset(pred_transform, 0, TREELITE_MAX_PRED_TRANSFORM_LENGTH * sizeof(char));
    std::strcpy(pred_transform, "identity");
  }
  ~ModelParam() = default;
  ModelParam(const ModelParam&) = default;
  ModelParam& operator=(const ModelParam&) = default;
  ModelParam(ModelParam&&) = default;
  ModelParam& operator=(ModelParam&&) = default;

  /* Override methods of dmlc::Parameter, since it can't handle char[] member */
  template<typename Container>
  inline std::vector<std::pair<std::string, std::string>>
  InitAllowUnknown(const Container &kwargs);
  template<typename Container>
  inline void UpdateDict(Container *dict) const;
  inline std::map<std::string, std::string> __DICT__() const;

  // declare parameters
  DMLC_DECLARE_PARAMETER(ModelParam) {
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
  ~Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

  void ReferenceSerialize(dmlc::Stream* fo) const;

  inline std::vector<PyBufferFrame> GetPyBuffer();
  inline void InitFromPyBuffer(std::vector<PyBufferFrame> frames);
};

/** Implementations **/

template<typename Container>
inline std::vector<std::pair<std::string, std::string> >
ModelParam::InitAllowUnknown(const Container& kwargs) {
  Container copy = kwargs;
  for (auto it = copy.begin(); it != copy.end(); ) {
    if (it->first == "pred_transform") {
      std::strncpy(this->pred_transform, it->second.c_str(),
                   TREELITE_MAX_PRED_TRANSFORM_LENGTH - 1);
      this->pred_transform[TREELITE_MAX_PRED_TRANSFORM_LENGTH - 1] = '\0';
      it = copy.erase(it);
    } else {
      ++it;
    }
  }
  return dmlc::Parameter<ModelParam>::InitAllowUnknown(copy);
}

template<typename Container>
inline void
ModelParam::UpdateDict(Container *dict) const {
  dmlc::Parameter<ModelParam>::UpdateDict(dict);
  (*dict)["pred_transform"] = std::string(this->pred_transform);
}

inline std::map<std::string, std::string>
ModelParam::__DICT__() const {
  auto ret = dmlc::Parameter<ModelParam>::__DICT__();
  ret.emplace("pred_transform", std::string(this->pred_transform));
  return ret;
}

inline PyBufferFrame GetPyBufferFromArray(void* data, const char* format,
                                                size_t itemsize, size_t nitem) {
  return PyBufferFrame{data, const_cast<char*>(format), itemsize, nitem};
}

// Infer format string from data type
template <typename T>
inline const char* InferFormatString() {
  switch (sizeof(T)) {
   case 1:
     return (std::is_unsigned<T>::value ? "=B" : "=b");
   case 2:
     return (std::is_unsigned<T>::value ? "=H" : "=h");
   case 4:
     if (std::is_integral<T>::value) {
       return (std::is_unsigned<T>::value ? "=L" : "=l");
     } else {
       CHECK(std::is_floating_point<T>::value);
       return "=f";
     }
   case 8:
     if (std::is_integral<T>::value) {
       return (std::is_unsigned<T>::value ? "=Q" : "=q");
     } else {
       CHECK(std::is_floating_point<T>::value);
       return "=d";
     }
   default:
     LOG(FATAL) << "Type not supported";
  }
  return nullptr;
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>& vec, const char* format) {
  return GetPyBufferFromArray(static_cast<void*>(vec.Data()), format, sizeof(T), vec.Size());
}

template <typename T>
inline PyBufferFrame GetPyBufferFromArray(ContiguousArray<T>& vec) {
  static_assert(std::is_arithmetic<T>::value,
      "Use GetPyBufferFromArray(vec, format) for composite types; specify format string manually");
  return GetPyBufferFromArray(vec, InferFormatString<T>());
}

inline PyBufferFrame GetPyBufferFromScalar(void* data, const char* format, size_t itemsize) {
  return GetPyBufferFromArray(data, format, itemsize, 1);
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T& scalar, const char* format) {
  return GetPyBufferFromScalar(static_cast<void*>(&scalar), format, sizeof(T));
}

template <typename T>
inline PyBufferFrame GetPyBufferFromScalar(T& scalar) {
  static_assert(std::is_arithmetic<T>::value,
      "Use GetPyBufferFromScalar(scalar, format) for composite types; "
      "specify format string manually");
  return GetPyBufferFromScalar(scalar, InferFormatString<T>());
}

template <typename T>
inline void InitArrayFromPyBuffer(ContiguousArray<T>& vec, PyBufferFrame buffer) {
  CHECK_EQ(sizeof(T), buffer.itemsize);
  vec.UseForeignBuffer(buffer.buf, buffer.nitem);
}

template <typename T>
inline void InitScalarFromPyBuffer(T& scalar, PyBufferFrame buffer) {
  CHECK_EQ(buffer.itemsize, sizeof(T));
  CHECK_EQ(buffer.nitem, 1);
  T* t = static_cast<T*>(buffer.buf);
  scalar = *t;
}

constexpr size_t kNumFramePerTree = 6;

inline std::vector<PyBufferFrame>
Tree::GetPyBuffer() {
  return {
    GetPyBufferFromScalar(num_nodes),
    GetPyBufferFromArray(nodes_, "T{=l=l=L=f=Q=d=d=b=b=?=?=?=?=H}"),
    GetPyBufferFromArray(leaf_vector_),
    GetPyBufferFromArray(leaf_vector_offset_),
    GetPyBufferFromArray(left_categories_),
    GetPyBufferFromArray(left_categories_offset_)
  };
}

inline void
Tree::InitFromPyBuffer(std::vector<PyBufferFrame> frames) {
  size_t frame_id = 0;
  InitScalarFromPyBuffer(num_nodes, frames[frame_id++]);
  InitArrayFromPyBuffer(nodes_, frames[frame_id++]);
  CHECK_EQ(num_nodes, nodes_.Size());
  InitArrayFromPyBuffer(leaf_vector_, frames[frame_id++]);
  InitArrayFromPyBuffer(leaf_vector_offset_, frames[frame_id++]);
  InitArrayFromPyBuffer(left_categories_, frames[frame_id++]);
  InitArrayFromPyBuffer(left_categories_offset_, frames[frame_id++]);
  CHECK_EQ(frame_id, kNumFramePerTree);
}

inline std::vector<PyBufferFrame>
Model::GetPyBuffer() {
  /* Header */
  std::vector<PyBufferFrame> frames{
    GetPyBufferFromScalar(num_feature),
    GetPyBufferFromScalar(num_output_group),
    GetPyBufferFromScalar(random_forest_flag),
    GetPyBufferFromScalar(param, "T{" _TREELITE_STR(TREELITE_MAX_PRED_TRANSFORM_LENGTH) "s=f=f}")
  };

  /* Body */
  for (auto& tree : trees) {
    auto tree_frames = tree.GetPyBuffer();
    frames.insert(frames.end(), tree_frames.begin(), tree_frames.end());
  }
  return frames;
}

inline void
Model::InitFromPyBuffer(std::vector<PyBufferFrame> frames) {
  /* Header */
  size_t frame_id = 0;
  InitScalarFromPyBuffer(num_feature, frames[frame_id++]);
  InitScalarFromPyBuffer(num_output_group, frames[frame_id++]);
  InitScalarFromPyBuffer(random_forest_flag, frames[frame_id++]);
  InitScalarFromPyBuffer(param, frames[frame_id++]);
  /* Body */
  const size_t num_frame = frames.size();
  CHECK_EQ((num_frame - frame_id) % kNumFramePerTree, 0);
  trees.clear();
  for (; frame_id < num_frame; frame_id += kNumFramePerTree) {
    std::vector<PyBufferFrame> tree_frames(frames.begin() + frame_id,
                                           frames.begin() + frame_id + kNumFramePerTree);
    trees.emplace_back();
    trees.back().InitFromPyBuffer(tree_frames);
  }
}

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
  CHECK_LE(nid, leaf_vector_offset_.Size());
  return std::vector<tl_float>(&leaf_vector_[leaf_vector_offset_[nid]],
                               &leaf_vector_[leaf_vector_offset_[nid + 1]]);
}

inline bool
Tree::HasLeafVector(int nid) const {
  CHECK_LE(nid, leaf_vector_offset_.Size());
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
  CHECK_LE(nid, left_categories_offset_.Size());
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

  const size_t end_oft = left_categories_offset_.Back();
  const size_t new_end_oft = end_oft + node_left_categories.size();
  CHECK_EQ(end_oft, left_categories_.Size());
  CHECK(std::all_of(&left_categories_offset_[nid + 1], left_categories_offset_.End(),
                    [end_oft](size_t x) { return (x == end_oft); }));
    // Hopefully we won't have to move any element as we add node_left_categories for node nid
  left_categories_.Extend(node_left_categories);
  CHECK_EQ(new_end_oft, left_categories_.Size());
  std::for_each(&left_categories_offset_[nid + 1], left_categories_offset_.End(),
                [new_end_oft](size_t& x) { x = new_end_oft; });
  std::sort(&left_categories_[end_oft], left_categories_.End());

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
  const size_t end_oft = leaf_vector_offset_.Back();
  const size_t new_end_oft = end_oft + node_leaf_vector.size();
  CHECK_EQ(end_oft, leaf_vector_.Size());
  CHECK(std::all_of(&leaf_vector_offset_[nid + 1], leaf_vector_offset_.End(),
                    [end_oft](size_t x) { return (x == end_oft); }));
    // Hopefully we won't have to move any element as we add leaf vector elements for node nid
  leaf_vector_.Extend(node_leaf_vector);
  CHECK_EQ(new_end_oft, leaf_vector_.Size());
  std::for_each(&leaf_vector_offset_[nid + 1], leaf_vector_offset_.End(),
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
