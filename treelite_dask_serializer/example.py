from .serializer import serialize, deserialize
from .builder import Node, Tree, ModelBuilder
from .treelite_model import get_frames
import numpy as np

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

def print_byte_ndarray(x):	
    for i, e in enumerate(x):	
        if (i + 1) % 48 == 1:	
            print('    ', end='')	
        print(f'{e:02X}', end='')	
        if (i + 1) % 48 == 0:	
            print()	
        elif (i + 1) % 16 == 0:	
            print('  ', end='')	
        else:	
            print(' ', end='')	
    if len(x) % 48 != 0:	
        print()

def test_round_trip(model):
    frames = [np.asarray(x) for x in get_frames(model)]
    print('Python buffer frames:')
    for frame_id, frame in enumerate(frames):
        if getattr(frame.dtype, 'names', None) is None:
            print(f'  * Frame {frame_id}: dtype {frame.dtype}, length {len(frame)}')
            if frame.dtype == np.uint8:
                print_byte_ndarray(frame)
            else:
                print(f'    {repr(frame)}')
        else:
            if len(frame.dtype.names) == 13:  # Node type
                print(f'  * Frame {frame_id}: dtype Node, length {len(frame)}')
                print('    (cleft_, cright_, sindex_, info_, data_count_, sum_hess_, gain_, ' +
                      'split_type_, cmp_, missing_category_to_zero_, data_count_present_, \n     ' +
                      'sum_hess_present_, gain_present_)')
            else:  # ModelParam type
                print(f'  * Frame {frame_id}: dtype ModelParam, length {len(frame)}')
                print('    (pred_transform, sigmoid_alpha, global_bias)')
            print('    array([')
            for e in frame:
                print(f'        {e}')
            print('    ])')
    print()

    s = serialize(model)
    print(f'Serialized model bytes ({len(s)} bytes):')
    print_bytes(s)

    model2 = deserialize(s)
    print(f'Deserialized model')
    s2 = serialize(model2)
    assert s == s2, f'len(s2) = {len(s2)}'
    print('Round-trip conversion preserved all bytes\n')

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
