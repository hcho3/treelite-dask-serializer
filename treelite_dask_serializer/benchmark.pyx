# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .builder import Node, Tree, ModelBuilder
from .serializer import get_frames, init_from_frames
from .treelite_frontend cimport *
from .treelite_model cimport *
from libcpp.memory cimport unique_ptr
from typing import Tuple, Dict, List, Union

import time
import os
import tempfile
import struct
import pickle
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

    cdef ModelBuilderHandle builder
    cdef TreeBuilderHandle tree
    TreeliteCreateModelBuilder(num_feature, 1, False, &builder)
    TreeliteCreateTreeBuilder(&tree)

    cdef int nid, left_child_key, right_child_key
    cdef int level, i
    for level in range(depth + 1):
        for i in range(2**level):
            nid = 2**level - 1 + i
            TreeliteTreeBuilderCreateNode(tree, nid)

    for level in range(depth + 1):
        for i in range(2**level):
            nid = 2**level - 1 + i
            if level == depth:
                TreeliteTreeBuilderSetLeafNode(tree, nid, 0.5)
            else:
                left_child_key = 2 * nid + 1
                right_child_key = 2 * nid + 2
                TreeliteTreeBuilderSetNumericalTestNode(tree, nid, level % 2, '<', 0.0, True,
                                                        left_child_key, right_child_key)
    TreeliteTreeBuilderSetRootNode(tree, 0)
    TreeliteModelBuilderInsertTree(builder, tree, -1)

    cdef ModelHandle model_handle
    TreeliteModelBuilderCommitModel(builder, &model_handle)
    model = make_model(model_handle)
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

def _round_trip_file(model : TreeliteModel):
    with tempfile.TemporaryDirectory() as tempdir:
        tstart = time.perf_counter()
        header, frames = serialize(model)
        print(f'Serialized model to Python buffer frames in {time.perf_counter() - tstart} sec')

        tstart = time.perf_counter()
        msg = {'header': header, 'frames': frames}
        with open(os.path.join(tempdir, 'model.bin'), 'wb') as f:
            h = pickle.dumps(header)
            f.write(struct.pack('=Q', len(h)))
            f.write(h)
            f.write(struct.pack('=Q', len(frames)))
            for frame in frames:
                b = bytes(frame)
                f.write(struct.pack('=Q', len(b)))
                f.write(b)
        print(f'Wrote model to disk in {time.perf_counter() - tstart} sec')

        tstart = time.perf_counter()
        frames = []
        with open(os.path.join(tempdir, 'model.bin'), 'rb') as f:
            h_sz = struct.unpack('=Q', f.read(8))[0]
            h = f.read(h_sz)
            header = pickle.loads(h)
            nframe = struct.unpack('=Q', f.read(8))[0]
            for frame_id in range(nframe):
                frame_sz = struct.unpack('=Q', f.read(8))[0]
                frame = f.read(frame_sz)
                frames.append(frame)
        print(f'Read model from disk in {time.perf_counter() - tstart} sec')

        tstart = time.perf_counter()
        received_model = deserialize(header, frames)
        print(f'Deserialized model from Python buffer frames in {time.perf_counter() - tstart} sec')

def main(round_trip=True, tcp=False, disk=False):
    ### Build full binary decision tree with depth 24
    ### Call C++ API directly for speed
    model = build_full_tree(num_feature=3, depth=24)

    if round_trip:
        if tcp:
            asyncio.run(_round_trip_tcp(model))
        elif disk:
            _round_trip_file(model)
        else:
            _round_trip_local(model)
