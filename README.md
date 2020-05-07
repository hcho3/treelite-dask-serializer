# (Experimental) Zero-copy efficient model serializer for decision trees, in conjunction with Dask

## Notable features
* Revised Treelite object representaion: now all tree fields are flat and can be serialized very efficiently
* Minimal port of Treelite: only model builder and definition are included here.
* Uses Cython to seamlessly integrate C++ code with Python
* Implements Python Buffer protocol, so that tree object can be serialized zero-copy.
* Implements a custom serializer hook to integrate with Dask.distributed.
* Sends the serialized Treelite model over TCP

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
See [`output.txt`](output.txt) for an example output.

## Treelite Dask serializer hook
See [`example.py`](treelite_dask_serializer/example.py) for details.
```python
@dask_serialize.register(TreeliteModel)
def serialize_treelite_model(x : TreeliteModel) -> Tuple[Dict, List[memoryview]]:
    frames = get_frames(x)
    header = {'format_str': [x.format.encode('utf-8') for x in frames],
              'itemsize': [x.itemsize for x in frames]}
    return header, frames

@dask_deserialize.register(TreeliteModel)
def deserialize_treelite_model(header : Dict, frames: List[Union[bytes, memoryview]]):
    return init_from_frames(frames, header['format_str'], header['itemsize'])
```

## Benchmark
Benchmark code: [`benchmark.pyx`](treelite_dask_serializer/benchmark.pyx).

The benchmark consists of building a full binary decision tree with depth 24. The tree contains
(2^24 - 1) nodes.

### How to run
```bash
# Build the tree only; do not serialize
python -c 'from treelite_dask_serializer.benchmark import main; main(round_trip=False)'

# Build the tree and perform round trip (serialize then deserialize) in memory
python -c 'from treelite_dask_serializer.benchmark import main; main(round_trip=True, tcp=False)'

# Build the tree and perform round trip (serialize then deserialize) via TCP.
# Use loopback interface (localhost)
python -c 'from treelite_dask_serializer.benchmark import main; main(round_trip=True, tcp=True)'
```

### Results
**System details**

* Ubuntu 18.04 LTS
* CPU: Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz, 6 cores with hyperthreading
* RAM: 2x 16GB DDR4 2133 MHz

**Run time**

In-memory:
```
Time to build model: 10.527289680001559 sec
Serialized model to Python buffer frames in 2.878298982977867e-05 sec
Deserialized model from Python buffer frames in 1.2042990420013666e-05 sec
```

TCP localhost:
```
Time to build model: 10.581047819985542 sec
Serialized model to Python buffer frames in 1.6157020581886172e-05 sec
Sent model to TCP in 2.0535116069950163 sec
Received model to TCP in 1.8430845960101578 sec
Deserialized model from Python buffer frames in 2.8103007934987545e-05 sec
```

Note that it is essentially free to convert between Treelite objects and Python buffer frames.

**Peak memory consumption**
Peak memory consumption is measured using `/usr/bin/time -v`. Converting between Treelite objects
and Python buffer frames costs no memory overhead. Sending the model over TCP incurs 1.6 GB extra
memory, probably because TCP uses send/receive buffer.

| | Memory used (MB) |
|--|--|
|Build model only | 7265 |
|Round trip in memory | 7265 |
|Round trip via TCP localhost | 8903 |
