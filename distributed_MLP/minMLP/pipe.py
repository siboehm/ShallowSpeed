from abc import ABC, abstractmethod
from enum import Enum, auto
import pandas as pd
import numpy as np
from mpi4py import MPI

from minMLP.models import Sequential


class PipelineInstruction(Enum):
    ZeroGrad = 0
    Forward = 1
    BackwardGradAccumulate = 2
    BackwardGradReduce = 3
    LoadMicroBatch = 4
    OptimizerStep = 5


class Schedule(ABC):
    def __init__(self, num_micro_batches, num_stages, stage_id):
        assert stage_id < num_stages
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
        yield [(PipelineInstruction.ZeroGrad, {})]
        for microbatch_id in range(self.num_micro_batches):
            cmds = [
                (PipelineInstruction.LoadMicroBatch, {"micro_batch_id": microbatch_id}),
                (PipelineInstruction.Forward, {}),
            ]
            if microbatch_id == self.num_micro_batches - 1:
                cmds.append((PipelineInstruction.BackwardGradReduce, {}))
            else:
                cmds.append((PipelineInstruction.BackwardGradAccumulate, {}))

            cmds.append((PipelineInstruction.OptimizerStep, {}))
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

    def __init__(self, save_dir, batch_size, mubatch_size):
        assert batch_size % mubatch_size == 0, "μBatchsize must divide batchsize!"
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.mubatch_size = mubatch_size

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

    def load_micro_batch(self, batch_id, micro_batch_id):
        assert batch_id < self.get_num_batches()
        assert micro_batch_id < self.get_num_mubatches()

        start_idx = batch_id * self.batch_size + micro_batch_id * self.mubatch_size
        end_idx = min(len(self.x_train), start_idx + self.mubatch_size)

        return self.x_train[start_idx:end_idx], self.y_train[start_idx:end_idx]

    def get_num_batches(self):
        return len(self) // self.batch_size

    def get_num_mubatches(self):
        return self.batch_size // self.mubatch_size


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
            for command, kwargs in commands:
                match command:
                    case PipelineInstruction.LoadMicroBatch:
                        self.load_micro_batch(batch_id, **kwargs)
                    case PipelineInstruction.Forward:
                        self.forward()
                    case PipelineInstruction.BackwardGradReduce:
                        self.backward_and_reduce()
                    case PipelineInstruction.BackwardGradAccumulate:
                        self.backward_accumulate()
                    case PipelineInstruction.OptimizerStep:
                        self.optimizer_step()
                    case PipelineInstruction.ZeroGrad:
                        self.zero_grad()
                    case _:
                        raise NotImplementedError(command)

    def load_micro_batch(self, batch_id, micro_batch_id):
        self.inputs_fw, self.inputs_bw = self.dataset.load_micro_batch(batch_id, micro_batch_id)

    def forward(self):
        self.outputs_fw = self.model.forward(self.inputs_fw)

    def backward_and_reduce(self):
        # hooks for AllReducing-ing the gradient across all dp_workers
        self.model.register_grad_hook(
            lambda param: backprop_allreduce_gradient(self.dp_comm, param)
        )
        self.model.register_post_grad_hook(backprop_block_for_comms)

        # regular backwards pass will trigger the AR hooks
        self.outputs_bw = self.model.backward(self.inputs_bw)

        self.model.reset_grad_hooks()
        self.model.reset_post_grad_hooks()

    def backward_accumulate(self):
        self.outputs_bw = self.model.backward(self.inputs_bw)

    def optimizer_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.model.zero_grad()

    def _is_last_stage(self):
        return self.stage_id == self.pipeline_depth - 1

