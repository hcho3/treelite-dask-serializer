#include <treelite/frontend.h>
#include <treelite/tree.h>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>

treelite::PyBufferFrame DuplicatePyBufferFrame(treelite::PyBufferFrame frame) {
  const size_t nbyte = frame.itemsize * frame.nitem;
  void* buf = std::malloc(nbyte);
  CHECK(buf);
  std::memcpy(buf, frame.buf, nbyte);
  return treelite::PyBufferFrame{buf, frame.format, frame.itemsize, frame.nitem};
}

int main(void) {
  treelite::frontend::ModelBuilder builder(2, 1, false);
  for (int i = 0; i < 2; ++i) {
    treelite::frontend::TreeBuilder tree;
    for (int i = 0; i < 7; ++i) {
      tree.CreateNode(i);
    }
    tree.SetNumericalTestNode(0, 0, "<", 0.0f, true, 1, 2);
    tree.SetCategoricalTestNode(1, 0, {0, 1}, true, 3, 4);
    tree.SetCategoricalTestNode(2, 1, {0}, true, 5, 6);
    tree.SetRootNode(0);
    tree.SetLeafNode(3, -2.0f);
    tree.SetLeafNode(4, 1.0f);
    tree.SetLeafNode(5, -1.0f);
    tree.SetLeafNode(6, 2.0f);
    builder.InsertTree(&tree, -1);
  }

  treelite::Model model;
  CHECK(builder.CommitModel(&model));

  CHECK_EQ(model.trees.size(), 2);

  std::string s, s2;
  {
    std::unique_ptr<dmlc::Stream> strm(new dmlc::MemoryStringStream(&s));
    model.Serialize(strm.get());
  }

  std::vector<treelite::PyBufferFrame> frames = model.GetPyBuffer();
  std::vector<treelite::PyBufferFrame> frames2;
  std::transform(frames.begin(), frames.end(), std::back_inserter(frames2),
      [](treelite::PyBufferFrame x) { return DuplicatePyBufferFrame(x); });

  treelite::Model model2;
  model2.InitFromPyBuffer(frames2);
  {
    std::unique_ptr<dmlc::Stream> strm(new dmlc::MemoryStringStream(&s2));
    model2.Serialize(strm.get());
  }
  CHECK_EQ(s, s2);

  return 0;
}
