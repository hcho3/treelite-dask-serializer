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
  class Node {
   public:
    Node(Tree* tree, int nid)
      : tree_(tree), nid_(nid), sindex_(0), missing_category_to_zero_(false) {}
    /*! \brief index of left child */
    inline int cleft() const {
      return this->cleft_;
    }
    /*! \brief index of right child */
    inline int cright() const {
      return this->cright_;
    }
    /*! \brief index of default child when feature is missing */
    inline int cdefault() const {
      return this->default_left() ? this->cleft() : this->cright();
    }
    /*! \brief feature index of split condition */
    inline unsigned split_index() const {
      return sindex_ & ((1U << 31) - 1U);
    }
    /*! \brief when feature is unknown, whether goes to left child */
    inline bool default_left() const {
      return (sindex_ >> 31) != 0;
    }
    /*! \brief whether current node is leaf node */
    inline bool is_leaf() const {
      return cleft_ == -1;
    }
    /*! \return get leaf value of leaf node */
    inline tl_float leaf_value() const {
      return (this->info_).leaf_value;
    }
    /*!
     * \return get leaf vector of leaf node; useful for multi-class
     * random forest classifier
     */
    inline std::vector<tl_float> leaf_vector() const {
      const auto& leaf_vector = this->tree_->leaf_vector_;
      const auto& offset = this->tree_->leaf_vector_offset_;
      CHECK_LE(this->nid_, offset.size());
      return std::vector<tl_float>(&leaf_vector[offset[this->nid_]],
                                   &leaf_vector[offset[this->nid_ + 1]]);
    }
    /*!
     * \return tests whether leaf node has a non-empty leaf vector
     */
    inline bool has_leaf_vector() const {
      const auto& offset = this->tree_->leaf_vector_offset_;
      CHECK_LE(this->nid_, offset.size());
      return offset[this->nid_] != offset[this->nid_ + 1];
    }
    /*! \return get threshold of the node */
    inline tl_float threshold() const {
      return (this->info_).threshold;
    }
    /*! \brief get parent of the node */
    inline int parent() const {
      return parent_ & ((1U << 31) - 1);
    }
    /*! \brief whether current node is left child */
    inline bool is_left_child() const {
      return (parent_ & (1U << 31)) != 0;
    }
    /*! \brief whether current node is root */
    inline bool is_root() const {
      return parent_ == -1;
    }
    /*! \brief get comparison operator */
    inline Operator comparison_op() const {
      return cmp_;
    }
    /*!
     * \brief Get list of all categories belonging to the left child node.
     * Categories not in this list will belong to the right child node.
     * Categories are integers ranging from 0 to (n-1), where n is the number of
     * categories in that particular feature.
     * This list is assumed to be in ascending order.
     */
    inline std::vector<uint32_t> left_categories() const {
      const auto& left_categories = this->tree_->left_categories_;
      const auto& offset = this->tree_->left_categories_offset_;
      CHECK_LE(this->nid_, offset.size());
      return std::vector<uint32_t>(&left_categories[offset[this->nid_]],
                                   &left_categories[offset[this->nid_ + 1]]);
    }
    /*! \brief get feature split type */
    inline SplitFeatureType split_type() const {
      return split_type_;
    }
    /*! \brief test whether this node has data count */
    inline bool has_data_count() const {
      return data_count_present_;
    }
    /*! \brief get data count */
    inline size_t data_count() const {
      return data_count_;
    }
    /*! \brief test whether this node has hessian sum */
    inline bool has_sum_hess() const {
      return sum_hess_present_;
    }
    /*! \brief get hessian sum */
    inline double sum_hess() const {
      return sum_hess_;
    }
    /*! \brief test whether this node has gain value */
    inline bool has_gain() const {
      return gain_present_;
    }
    /*! \brief get gain value */
    inline double gain() const {
      return gain_;
    }
    /*! \brief test whether missing values should be converted into zero;
     *         only applicable for categorical splits */
    inline bool missing_category_to_zero() const {
      return missing_category_to_zero_;
    }
    /*!
     * \brief create a numerical split
     * \param split_index feature index to split
     * \param threshold threshold value
     * \param default_left the default direction when feature is unknown
     * \param cmp comparison operator to compare between feature value and
     *            threshold
     */
    inline void set_numerical_split(unsigned split_index, tl_float threshold,
                                    bool default_left, Operator cmp) {
      CHECK_LT(split_index, (1U << 31) - 1) << "split_index too big";
      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      (this->info_).threshold = threshold;
      this->cmp_ = cmp;
      this->split_type_ = SplitFeatureType::kNumerical;
    }
    /*!
     * \brief create a categorical split
     * \param split_index feature index to split
     * \param threshold threshold value
     * \param default_left the default direction when feature is unknown
     * \param cmp comparison operator to compare between feature value and
     *            threshold
     */
    inline void set_categorical_split(unsigned split_index, bool default_left,
                                      bool missing_category_to_zero,
                                      const std::vector<uint32_t>& left_categories) {
      CHECK_LT(split_index, (1U << 31) - 1) << "split_index too big";

      auto& t_left_categories = this->tree_->left_categories_;
      auto& t_left_categories_offset = this->tree_->left_categories_offset_;
      const size_t end_oft = t_left_categories_offset.back();
      const size_t new_end_oft = end_oft + left_categories.size();
      CHECK_EQ(end_oft, t_left_categories.size());
      CHECK(std::all_of(t_left_categories_offset.begin() + (this->nid_ + 1),
                        t_left_categories_offset.end(),
                        [end_oft](size_t x) { return (x == end_oft); }));
        // Hopefully we won't have to move any element as we add left_categories for node nid
      t_left_categories.insert(t_left_categories.end(), left_categories.begin(),
                               left_categories.end());
      CHECK_EQ(new_end_oft, t_left_categories.size());
      std::for_each(t_left_categories_offset.begin() + (this->nid_ + 1),
                    t_left_categories_offset.end(),
                    [new_end_oft](size_t& x) { x = new_end_oft; });
      std::sort(t_left_categories.begin() + end_oft, t_left_categories.end());

      if (default_left) split_index |= (1U << 31);
      this->sindex_ = split_index;
      this->split_type_ = SplitFeatureType::kCategorical;
      this->missing_category_to_zero_ = missing_category_to_zero;
    }
    /*!
     * \brief set the leaf value of the node
     * \param value leaf value
     */
    inline void set_leaf(tl_float value) {
      (this->info_).leaf_value = value;
      this->cleft_ = -1;
      this->cright_ = -1;
      this->split_type_ = SplitFeatureType::kNone;
    }
    /*!
     * \brief set the leaf vector of the node; useful for multi-class
     * random forest classifier
     * \param leaf_vector leaf vector
     */
    inline void set_leaf_vector(const std::vector<tl_float>& leaf_vector) {
			auto& t_leaf_vector = this->tree_->leaf_vector_;
			auto& t_leaf_vector_offset = this->tree_->leaf_vector_offset_;
      const size_t end_oft = t_leaf_vector_offset.back();
      const size_t new_end_oft = end_oft + leaf_vector.size();
      CHECK_EQ(end_oft, t_leaf_vector.size());
      CHECK(std::all_of(t_leaf_vector_offset.begin() + (this->nid_ + 1),
                        t_leaf_vector_offset.end(),
                        [end_oft](size_t x) { return (x == end_oft); }));
        // Hopefully we won't have to move any element as we add leaf vector elements for node nid
      t_leaf_vector.insert(t_leaf_vector.end(), leaf_vector.begin(), leaf_vector.end());
      CHECK_EQ(new_end_oft, t_leaf_vector.size());
      std::for_each(t_leaf_vector_offset.begin() + (this->nid_ + 1),
                    t_leaf_vector_offset.end(),
                    [new_end_oft](size_t& x) { x = new_end_oft; });

      this->cleft_ = -1;
      this->cright_ = -1;
      this->split_type_ = SplitFeatureType::kNone;
    }
    /*!
     * \brief set the hessian sum of the node
     * \param sum_hess hessian sum
     */
    inline void set_sum_hess(double sum_hess) {
      this->sum_hess_ = sum_hess;
    }
    /*!
     * \brief set the data count of the node
     * \param data_count data count
     */
    inline void set_data_count(size_t data_count) {
      this->data_count_ = data_count;
    }
    /*!
     * \brief set the gain value of the node
     * \param gain gain value
     */
    inline void set_gain(double gain) {
      this->gain_ = gain;
    }
    /*!
     * \brief set parent of the node
     * \param pidx node id of the parent
     * \param is_left_child whether the node is left child or not
     */
    inline void set_parent(int pidx, bool is_left_child = true) {
      if (is_left_child) pidx |= (1U << 31);
      this->parent_ = pidx;
    }

   private:
    friend class Tree;
    /*! \brief pointer to tree the node belongs to */
    Tree* tree_;
    /*! \brief node ID of this node, if tree_ is not nullptr */
    int nid_;
    /*! \brief store either leaf value or decision threshold */
    union Info {
      tl_float leaf_value;  // for leaf nodes
      tl_float threshold;   // for non-leaf nodes
    };
    /*!
     * \brief pointer to parent
     * highest bit is used to indicate whether it's a left child or not
     */
    int parent_;
    /*! \brief pointer to left and right children */
    int cleft_, cright_;
    /*! \brief feature split type */
    SplitFeatureType split_type_;
    /*!
     * \brief feature index used for the split
     * highest bit indicates default direction for missing values
     */
    unsigned sindex_;
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
    size_t data_count_;
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
      nodes_.emplace_back(this, nid);
    }
    return nd;
  }

 public:
  /*! \brief number of nodes */
  int num_nodes;
  /*!
   * \brief get node given nid
   * \param nid node id
   * \return reference to node
   */
  inline Node& operator[](int nid) {
    return nodes_[nid];
  }
  /*!
   * \brief get node given nid (const version)
   * \param nid node id
   * \return const reference to node
   */
  inline const Node& operator[](int nid) const {
    return nodes_[nid];
  }
  /*! \brief initialize the model with a single root node */
  inline void Init() {
    num_nodes = 1;
    leaf_vector_.clear();
    leaf_vector_offset_ = {0, 0};
    left_categories_.clear();
    left_categories_offset_ = {0, 0};
    nodes_.emplace_back(this, 0);
    nodes_[0].set_leaf(0.0f);
    nodes_[0].set_parent(-1);
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
    nodes_[cleft].set_parent(nid, true);
    nodes_[cright].set_parent(nid, false);
  }

  /*!
   * \brief get list of all categorical features that have appeared anywhere
   *        in tree
   */
  inline std::vector<unsigned> GetCategoricalFeatures() const {
    std::unordered_map<unsigned, bool> tmp;
    for (int nid = 0; nid < num_nodes; ++nid) {
      const Node& node = nodes_[nid];
      const SplitFeatureType type = node.split_type();
      if (type != SplitFeatureType::kNone) {
        const bool flag = (type == SplitFeatureType::kCategorical);
        const unsigned split_index = node.split_index();
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

}  // namespace treelite
#endif  // TREELITE_TREE_H_
