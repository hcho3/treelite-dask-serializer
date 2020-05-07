from .serializer import serialize as treelite2bytes  # for testing purposes only
from .builder import Node, Tree, ModelBuilder
from .treelite_model import get_frames, init_from_frames, TreeliteModel
import numpy as np
from distributed.protocol import dask_serialize, dask_deserialize, serialize, deserialize
from typing import Tuple, Dict, List

def print_bytes(s):
    for i, e in enumerate(s):
        if (i + 1) % 48 == 1:
            print('    ', end='')
        print(f'{e:02X}', end='')
        if (i + 1) % 48 == 0:
            print()
        elif (i + 1) % 16 == 0:
            print('  ', end='')
        else:
            print(' ', end='')
    if len(s) % 48 != 0:
        print()

def print_buffer_frames(frames : List[memoryview]):
    for frame_id, frame in enumerate(frames):
        _frame = np.asarray(frame)
        if getattr(_frame.dtype, 'names', None) is None:
            print(f'  * Frame {frame_id}: dtype {_frame.dtype}, length {len(_frame)}')
            print(f'    {repr(_frame)}')
        else:
            if len(_frame.dtype.names) == 14:  # Node type
                print(f'  * Frame {frame_id}: dtype Node, length {len(_frame)}')
                print('    (cleft_, cright_, sindex_, info_, data_count_, sum_hess_, gain_, ' +
                      'split_type_, cmp_, missing_category_to_zero_, data_count_present_, \n     ' +
                      'sum_hess_present_, gain_present_, pad_)')
            else:  # ModelParam type
                print(f'  * Frame {frame_id}: dtype ModelParam, length {len(_frame)}')
                print('    (pred_transform, sigmoid_alpha, global_bias)')
            print('    array([')
            for e in _frame:
                print(f'        {e}')
            print('    ])')
    print()

@dask_serialize.register(TreeliteModel)
def serialize_treelite_model(x : TreeliteModel) -> Tuple[Dict, List[memoryview]]:
    header = {}
    frames = get_frames(x)
    return header, frames

@dask_deserialize.register(TreeliteModel)
def deserialize_treelite_model(header : Dict, frames: List[memoryview]):
    return init_from_frames(frames)

def test_round_trip(model : TreeliteModel):
    header, frames = serialize(model)

    print('Serialized model to Python buffer frames:')
    print_buffer_frames(frames)

    result = deserialize(header, frames)
    print(f'Deserialized model from Python buffer frames')

    assert treelite2bytes(model) == treelite2bytes(result)
    print('Round-trip serialization (via Python buffer) preserved all bytes\n')

def tree_stump():
    builder = ModelBuilder(num_feature=2)

    tree = Tree()
    tree[0].set_numerical_test_node(feature_id=0, opname='<', threshold=0, default_left=True,
            left_child_key=1, right_child_key=2)
    tree[0].set_root()
    tree[1].set_leaf_node(-1)
    tree[2].set_leaf_node(1)
    builder.append(tree)

    model = builder.commit()
    print('Built a tree stump')
    test_round_trip(model)

def tree_stump_leaf_vec():
    builder = ModelBuilder(num_feature=2, num_output_group=2, random_forest=True)

    tree = Tree()
    tree[0].set_numerical_test_node(feature_id=0, opname='<', threshold=0, default_left=True,
            left_child_key=1, right_child_key=2)
    tree[0].set_root()
    tree[1].set_leaf_node([-1, 1])
    tree[2].set_leaf_node([1, -1])
    builder.append(tree)

    model = builder.commit()
    print('Built a tree stump with leaf vector')
    test_round_trip(model)

def tree_stump_categorical_split():
    builder = ModelBuilder(num_feature=2)

    tree = Tree()
    tree[0].set_categorical_test_node(feature_id=0, left_categories=[0, 1], default_left=True,
            left_child_key=1, right_child_key=2)
    tree[0].set_root()
    tree[1].set_leaf_node(-1)
    tree[2].set_leaf_node(1)

    builder.append(tree)

    model = builder.commit()
    print('Built a tree stump with a categorical split')
    test_round_trip(model)

def tree_depth2():
    builder = ModelBuilder(num_feature=2, pred_transform='sigmoid', global_bias=0.5)

    for _ in range(2):
        tree = Tree()
        tree[0].set_numerical_test_node(feature_id=0, opname='<', threshold=0, default_left=True,
                left_child_key=1, right_child_key=2)
        tree[1].set_categorical_test_node(feature_id=0, left_categories=[0, 1], default_left=True,
                left_child_key=3, right_child_key=4)
        tree[2].set_categorical_test_node(feature_id=1, left_categories=[0], default_left=True,
                left_child_key=5, right_child_key=6)
        tree[0].set_root()
        tree[3].set_leaf_node(-2)
        tree[4].set_leaf_node(1)
        tree[5].set_leaf_node(-1)
        tree[6].set_leaf_node(2)
        builder.append(tree)

    model = builder.commit()
    print('Built 2 trees with depth 2, mix of categorical and numerical splits')
    test_round_trip(model)

def main():
    tree_stump()
    tree_stump_leaf_vec()
    tree_stump_categorical_split()
    tree_depth2()
