# (Experimental) Zero-copy efficient model serializer for decision trees, in conjunction with Dask

## Notable features
* Revised Treelite object representaion: now all tree fields are flat and can be serialized very efficiently
* Minimal port of Treelite: only model builder and definition are included here.
* Uses Cython to seamlessly integrate C++ code with Python
* Implements Python Buffer protocol, so that tree object can be serialized zero-copy.
* Implements a custom serializer hook to integrate with Dask.distributed.

## TODOs
* Run benchmark with TCP

## How to build
```bash
# Build libtreelite.a
mkdir -p treelite_minimal/build/
cd treelite_minimal/build/
cmake ..
make -j

cd ../..
# Build Python extension (written in Cython)
python setup.py build_ext --inplace
# Run example
python -c 'from treelite_dask_serializer.example import main; main()'
```
See `output.txt` for an example output.

## Treelite Dask serializer hook
See the full codebase for details.
```python
@dask_serialize.register(TreeliteModel)
def serialize_treelite_model(x : TreeliteModel) -> Tuple[Dict, List[memoryview]]:
    header = {}
    frames = get_frames(x)
    return header, frames

@dask_deserialize.register(TreeliteModel)
def deserialize_treelite_model(header : Dict, frames: List[memoryview]):
    return init_from_frames(frames)
```
