from hashlib import sha1

import torch
from minMLP.base import Parameter
from mpi4py import MPI


def rprint(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)


def get_model_hash(model):
    # this is probably not the most efficient way to do this, but it's
    # not straightforward to get a deterministic, content-based hash of a model's parameters
    hash_str = ""
    for param in model.parameters():
        if torch.is_tensor(param):
            param = param.data.cpu().numpy()
        if isinstance(param, Parameter):
            param = param.data

        # concat the strings to form a single hash later
        hash_str += sha1(param).hexdigest()
    # hash to concatenated strings
    return sha1(hash_str.encode("utf-8")).hexdigest()


def assert_sync(comm, model_hash):
    # check that all processes have the same model hash
    model_hash_all = comm.gather(model_hash, root=0)
    if comm.rank == 0 and len(set(model_hash_all)) > 1:
        raise ValueError("Model hash mismatch")
