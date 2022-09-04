from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from mpi4py import MPI

from minMLP.dataset import Dataset
from minMLP.layers import Sequential


class PipeInstr(Enum):
    ZeroGrad = 0
    Forward = 1
    BackwardGradAccumulate = 2
    BackwardGradAllReduce = 3
    LoadMicroBatchInput = 4
    LoadMicroBatchTarget = 5
    OptimizerStep = 6
    ReceiveActivations = 7
    SendActivations = 8
    ReceiveOutputGrad = 9
    SendInputGrad = 10


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

    @abstractmethod
    def num_buffers(self):
        """
        The number of buffers necessary for sending & receiving data
        """
        pass

    @property
    def is_last_stage(self):
        return self.stage_id == self.num_stages - 1

    @property
    def is_first_stage(self):
        return self.stage_id == 0

    def is_valid_stage_id(self, stage_id):
        return 0 <= stage_id < self.num_stages


class DataParallelSchedule(Schedule):
    """
    A pure data parallel schedule, without any proper pipeline parallelism
    """

    def steps(self):
        # naive pipeline parallel, hence we only need two buffers
        left_buf = 0
        right_buf = 1

        yield [(PipeInstr.ZeroGrad, {})]

        for mubatch_id in range(self.num_micro_batches):
            cmds = []

            if self.is_first_stage:
                cmds.append(
                    (
                        PipeInstr.LoadMicroBatchInput,
                        {"mubatch_id": mubatch_id, "buffer_idx": left_buf},
                    )
                )
            else:
                cmds.append((PipeInstr.ReceiveActivations, {"buffer_idx": left_buf}))

            cmds.append((PipeInstr.Forward, {"buffer_idx": left_buf}))

            if self.is_last_stage:
                cmds.append(
                    (
                        PipeInstr.LoadMicroBatchTarget,
                        {"mubatch_id": mubatch_id, "buffer_idx": right_buf},
                    )
                )
            else:
                cmds.append((PipeInstr.SendActivations, {"buffer_idx": right_buf}))
                cmds.append((PipeInstr.ReceiveOutputGrad, {"buffer_idx": right_buf}))

            if mubatch_id == self.num_micro_batches - 1:
                cmds.append(
                    (PipeInstr.BackwardGradAllReduce, {"buffer_idx": right_buf})
                )
            else:
                cmds.append(
                    (PipeInstr.BackwardGradAccumulate, {"buffer_idx": right_buf})
                )

            if not self.is_first_stage:
                cmds.append((PipeInstr.SendInputGrad, {"buffer_idx": left_buf}))

            if mubatch_id == self.num_micro_batches - 1:
                cmds.append((PipeInstr.OptimizerStep, {}))
            yield cmds

    def num_buffers(self):
        # need 1 Buffer for receiving input and 1 buffer for sending output
        # since this is naive PP, there's only ever one μB in flight at the
        # same time
        return 2


class InferenceSchedule(Schedule):
    def steps(self):
        left_buf = 0
        right_buf = 1

        for mubatch_id in range(self.num_micro_batches):
            cmds = []

            if self.is_first_stage:
                cmds.append(
                    (
                        PipeInstr.LoadMicroBatchInput,
                        {"mubatch_id": mubatch_id, "buffer_idx": left_buf},
                    )
                )
            else:
                cmds.append((PipeInstr.ReceiveActivations, {"buffer_idx": left_buf}))

            cmds.append((PipeInstr.Forward, {"buffer_idx": left_buf}))

            if not self.is_last_stage:
                cmds.append((PipeInstr.SendActivations, {"buffer_idx": right_buf}))
            yield cmds

    def num_buffers(self):
        return 2


class GPipeSchedule:
    def __init__(self):
        pass


class PipeDreamSchedule:
    def __init__(self):
        pass


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

    def __init__(
        self,
        dp_comm: MPI.Comm,
        pp_comm: MPI.Comm,
        model: Sequential,
        dataset: Dataset,
        optimizer,
    ):
        self.stage_id = pp_comm.Get_rank()
        self.pipeline_depth = pp_comm.Get_size()
        self.dp_comm = dp_comm
        self.pp_comm = pp_comm
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.buffers = []

    def execute(self, sched, batch_id):
        """
        Use the configured schedule to execute a full batch

        Basically it'll just call the right function, given whatever the scheduler
        tells it to do
        """

        # bufferization
        self.buffers = [None for _ in range(sched.num_buffers())]
        self.buffers[0] = np.empty((self.dataset.mubatch_size, self.model.in_dim))

        for commands in sched.steps():
            for command, kwargs in commands:
                match command:
                    case PipeInstr.LoadMicroBatchInput:
                        self.load_micro_batch_input(batch_id, **kwargs)
                    case PipeInstr.LoadMicroBatchTarget:
                        self.load_micro_batch_target(batch_id, **kwargs)
                    case PipeInstr.Forward:
                        self.forward(**kwargs)
                    case PipeInstr.BackwardGradAllReduce:
                        self.backward_and_reduce(**kwargs)
                    case PipeInstr.BackwardGradAccumulate:
                        self.backward_accumulate(**kwargs)
                    case PipeInstr.OptimizerStep:
                        self.optimizer_step()
                    case PipeInstr.ZeroGrad:
                        self.zero_grad()
                    case PipeInstr.ReceiveActivations:
                        self.recv_buffer(from_idx=self.get_predecessor(sched), **kwargs)
                    case PipeInstr.SendActivations:
                        self.send_buffer(to_idx=self.get_successor(sched), **kwargs)
                    case PipeInstr.ReceiveOutputGrad:
                        self.recv_buffer(from_idx=self.get_successor(sched), **kwargs)
                    case PipeInstr.SendInputGrad:
                        self.send_buffer(to_idx=self.get_predecessor(sched), **kwargs)
                    case _:
                        raise NotImplementedError(command)

    def load_micro_batch_input(self, batch_id, mubatch_id, buffer_idx):
        self.buffers[buffer_idx] = self.dataset.load_micro_batch_input(
            batch_id, mubatch_id
        )

    def get_predecessor(self, schedule):
        pred = self.stage_id - 1
        assert schedule.is_valid_stage_id(pred)
        return pred

    def get_successor(self, schedule):
        succ = self.stage_id + 1
        assert schedule.is_valid_stage_id(succ)
        return succ

    def recv_buffer(self, from_idx, buffer_idx):
        self.pp_comm.Recv(self.buffers[buffer_idx], from_idx)

    def send_buffer(self, to_idx, buffer_idx):
        self.pp_comm.Send(self.buffers[buffer_idx], to_idx)

    def load_micro_batch_target(self, batch_id, mubatch_id, buffer_idx):
        self.buffers[buffer_idx] = self.dataset.load_micro_batch_target(
            batch_id, mubatch_id
        )

    def forward(self, buffer_idx):
        self.buffers[buffer_idx + 1] = self.model.forward(self.buffers[buffer_idx])

    def backward_and_reduce(self, buffer_idx):
        # hooks for AllReducing-ing the gradient across all dp_workers
        self.model.register_grad_hook(
            lambda param: backprop_allreduce_gradient(self.dp_comm, param)
        )
        self.model.register_post_grad_hook(backprop_block_for_comms)

        # regular backwards pass will trigger the AR hooks
        self.backward_accumulate(buffer_idx)

        self.model.reset_grad_hooks()
        self.model.reset_post_grad_hooks()

    def backward_accumulate(self, buffer_idx):
        self.buffers[buffer_idx - 1] = self.model.backward(self.buffers[buffer_idx])

    def optimizer_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.model.zero_grad()
