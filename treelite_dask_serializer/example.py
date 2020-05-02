from .serializer import serialize, deserialize
from .builder import Node, Tree, ModelBuilder

def print_bytes(s):
    for i, e in enumerate(s):
        print(f'{e:02X}', end='')
        if (i + 1) % 48 == 0:
            print()
        elif (i + 1) % 16 == 0:
            print('  ', end='')
        elif (i + 1) % 2 == 0:
            print(' ', end='')
    if len(s) % 16 != 0:
        print()

def test_round_trip(model):
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
    builder = ModelBuilder(num_feature=2)

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
    print('Built a tree with depth 2, mix of categorical and numerical splits')
    test_round_trip(model)

def main():
    tree_stump()
    tree_stump_leaf_vec()
    tree_stump_categorical_split()
    tree_depth2()
