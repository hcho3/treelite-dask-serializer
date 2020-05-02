# (Experimental) Zero-copy efficient model serializer for decision trees, in conjunction with Dask

## Notable features
* Revised Treelite object representaion: now all tree fields are flat and can be serialized very efficiently
* Minimal port of Treelite: only model builder and definition are included here.
* Cython, to seamlessly integrate C++ code with Python

## WIP
* Implement Python Buffer protocol, so that tree object can be serialized zero-copy. (Right now one copy is made.)

## TODOs
* Integrate with Dask via a custom serializer hook.

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
