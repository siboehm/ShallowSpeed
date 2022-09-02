from abc import ABC, abstractmethod
from enum import Enum, auto
import pandas as pd
import numpy as np
from mpi4py import MPI

from minMLP.models import Sequential


class PipelineInstruction(Enum):
    Forward = 1
    Backward = 2
    BackwardAndReduce = 3
    LoadMicroBatch = 4
    OptimizerStep = 5


class Schedule(ABC):
    def __init__(self, num_micro_batches, num_stages, stage_id):
        self.num_stages = num_stages
        self.stage_id = stage_id
        self.num_micro_batches = num_micro_batches

    @abstractmethod
    def steps(self):
        """
        This returns a generator, which contains all the operations to execute
        for processing a single batch from beginning to end
        """
        pass


class DataParallelSchedule(Schedule):
    """
    A pure data parallel schedule, without any proper pipeline parallelism
    """

    def steps(self):
        for microbatch_id in range(self.num_micro_batches):
            cmds = [
                PipelineInstruction.LoadMicroBatch,
                PipelineInstruction.Forward,
            ]
            if microbatch_id == self.num_micro_batches - 1:
                cmds.append(PipelineInstruction.BackwardAndReduce)
            else:
                cmds.append(PipelineInstruction.Backward)

            cmds.append(PipelineInstruction.OptimizerStep)
            yield cmds


class GPipeSchedule:
    def __init__(self):
        pass


class PipeDreamSchedule:
    def __init__(self):
        pass


class Dataset:
    x_train = None
    y_train = None

    def __init__(self, save_dir, batch_size):
        self.save_dir = save_dir
        self.batch_size = batch_size

    def load(self, rank, size):
        # each process loads the whole dataset
        # this is inefficient for large datasets, but fine for tiny MNIST
        x_train = pd.read_parquet(self.save_dir / "x_train.parquet").to_numpy()
        y_train = np.load(self.save_dir / "y_train.npy")

        # each process selects its subset of the datasets by a `rank`-offset and `size`-strides
        # the copy() is super important, else the array is not continuous in memory
        # which results in horrible matmul performance
        self.x_train = x_train[rank : len(x_train) : size].copy()
        self.y_train = y_train[rank : len(y_train) : size].copy()

    def __len__(self):
        return len(self.x_train)

    def load_batch(self, batch_id):
        assert batch_id < self.get_num_batches()

        start_idx = batch_id * self.batch_size
        end_idx = min(len(self.x_train), batch_id * self.batch_size + self.batch_size)

        x = self.x_train[start_idx:end_idx]
        y = self.y_train[start_idx:end_idx]
        return x, y

    def get_num_batches(self):
        return len(self.x_train) // self.batch_size


def backprop_allreduce_gradient(comm, param):
    # start a non-blocking AllReduce for the parameters for which we just
    # calculated the final gradient.
    # This interleaves communication of this layer's gradients with
    # computation of the next layers gradients
    if param.requires_grad:
        param._request = comm.Iallreduce(
            sendbuf=MPI.IN_PLACE, recvbuf=param.grad, op=MPI.SUM
        )


def backprop_block_for_comms(params):
    # after the full backwards pass we wait for all communication to finish
    # only then can we be certain that the gradients are the same on all processes
    requests = [
        param._request
        for param in params
        if param.requires_grad and param._request is not None
    ]
    MPI.Request.Waitall(requests)


class Worker:
    """
    Executes all stages in a schedule, during each batch
    There is not state kept across batches

    but within the microbatches we need to keep state, mainly the gradients,
    but also the buffers for sending and receiving activations & gradients
    """

    def __init__(self, dp_comm, pp_comm, model: Sequential, dataset, optimizer):
        self.stage_id = pp_comm.Get_rank()
        self.pipeline_depth = pp_comm.Get_size()
        self.dp_comm = dp_comm
        self.pp_comm = pp_comm
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer

        # allocate buffers for current microbatch
        self.inputs_fw = None
        self.outputs_fw = None
        self.inputs_bw = None
        self.outputs_bw = None

    def execute(self, schedule, batch_id):
        """
        Use the configured schedule to execute a full batch

        Basically it'll just call the right function, given whatever the scheduler
        tells it to do
        """
        for commands in schedule.steps():
            for command in commands:
                Worker.INSTR_MAP[command](self, batch_id=batch_id)

    def load_micro_batch(self, **kwargs):
        self.inputs_fw, self.inputs_bw = self.dataset.load_batch(kwargs["batch_id"])

    def forward(self, **kwargs):
        self.outputs_fw = self.model.forward(self.inputs_fw)

    def backward_and_reduce(self, **kwargs):
        # hooks for AllReducing-ing the gradient across all dp_workers
        self.model.register_grad_hook(
            "allreduce", lambda param: backprop_allreduce_gradient(self.dp_comm, param)
        )
        self.model.register_post_grad_hook("block", backprop_block_for_comms)

        self.outputs_bw = self.model.backward(self.inputs_bw)

        self.model.unregister_grad_hook("allreduce")
        self.model.unregister_post_grad_hook("block")

    def optimizer_step(self, **kwargs):
        self.optimizer.step()

    INSTR_MAP = {
        PipelineInstruction.LoadMicroBatch: load_micro_batch,
        PipelineInstruction.Forward: forward,
        PipelineInstruction.BackwardAndReduce: backward_and_reduce,
        PipelineInstruction.OptimizerStep: optimizer_step,
    }

    def _is_last_stage(self):
        return self.stage_id == self.pipeline_depth - 1
