# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .builder import Node, Tree, ModelBuilder
from .serializer import get_frames, init_from_frames
from .treelite_frontend cimport TreeBuilder, ModelBuilder
from .treelite_model cimport make_model, TreeliteModel, Model as NativeModel
from libcpp.memory cimport unique_ptr
from typing import Tuple, Dict, List, Union
import time
import asyncio

from distributed.comm import connect, listen

def serialize(model : TreeliteModel) -> Tuple[Dict, List[memoryview]]:
    frames = get_frames(model)
    header = {'format_str': [x.format.encode('utf-8') for x in frames],
              'itemsize': [x.itemsize for x in frames]}
    return header, frames

def deserialize(header : Dict, frames: List[Union[bytes, memoryview]]) -> TreeliteModel:
    return init_from_frames(frames, header['format_str'], header['itemsize'])

def build_full_tree(num_feature : int, depth : int) -> TreeliteModel:
    tstart = time.perf_counter()

    cdef unique_ptr[ModelBuilder] builder
    cdef unique_ptr[TreeBuilder] tree
    builder.reset(new ModelBuilder(num_feature, 1, False))
    tree.reset(new TreeBuilder())

    cdef int nid, left_child_key, right_child_key
    cdef int level, i
    for level in range(depth + 1):
        for i in range(2**level):
            nid = 2**level - 1 + i
            tree.get().CreateNode(nid)

    for level in range(depth + 1):
        for i in range(2**level):
            nid = 2**level - 1 + i
            if level == depth:
                tree.get().SetLeafNode(nid, 0.5)
            else:
                left_child_key = 2 * nid + 1
                right_child_key = 2 * nid + 2
                tree.get().SetNumericalTestNode(nid, level % 2, '<', 0.0, True,
                                                left_child_key, right_child_key)
    tree.get().SetRootNode(0)
    builder.get().InsertTree(tree.get(), -1)

    cdef unique_ptr[NativeModel] model_handle
    model_handle.reset(new NativeModel())
    builder.get().CommitModel(model_handle.get())
    model = make_model(model_handle.release())
    print(f'Time to build model: {time.perf_counter() - tstart} sec')

    return model

def _round_trip_local(model : TreeliteModel):
    tstart = time.perf_counter()
    header, frames = serialize(model)
    print(f'Serialized model to Python buffer frames in {time.perf_counter() - tstart} sec')

    tstart = time.perf_counter()
    received_model = deserialize(header, frames)
    print(f'Deserialized model from Python buffer frames in {time.perf_counter() - tstart} sec')

async def get_comm_pair(listen_addr, listen_args={}, connect_args={}, **kwargs):
    q = asyncio.Queue()

    async def handle_comm(comm):
        await q.put(comm)

    listener = await listen(listen_addr, handle_comm, **listen_args, **kwargs)
    comm = await connect(listener.contact_address, **connect_args, **kwargs)
    serv_comm = await q.get()
    return (comm, serv_comm)

async def _client_loop(comm, model : TreeliteModel):
    tstart = time.perf_counter()
    header, frames = serialize(model)
    print(f'Serialized model to Python buffer frames in {time.perf_counter() - tstart} sec')
    tstart = time.perf_counter()
    msg = (header, frames)
    await comm.write(msg)
    print(f'Sent model to TCP in {time.perf_counter() - tstart} sec')

async def _server_loop(comm, model : TreeliteModel):
    tstart = time.perf_counter()
    received_msg = await comm.read()
    print(f'Received model to TCP in {time.perf_counter() - tstart} sec')
    tstart = time.perf_counter()
    header, frames = received_msg
    received_model = deserialize(header, frames)
    print(f'Deserialized model from Python buffer frames in {time.perf_counter() - tstart} sec')

async def _round_trip_tcp(model : TreeliteModel):
    client, server = await get_comm_pair('tcp://localhost')
    await asyncio.gather(_client_loop(client, model), _server_loop(server, model))
    await client.close()
    await server.close()

def main(round_trip=True, tcp=False):
    ### Build full binary decision tree with depth 24
    ### Call C++ API directly for speed
    model = build_full_tree(num_feature=3, depth=24)

    if round_trip:
        if tcp:
            asyncio.run(_round_trip_tcp(model))
        else:
            _round_trip_local(model)
