from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from mpi4py import MPI

from minMLP.dataset import Dataset
from minMLP.layers import MLP


class PipeInstr(Enum):
    ZeroGrad = 0
    Forward = 1
    BackwardGradAccumulate = 2
    BackwardGradAllReduce = 3
    LoadMicroBatchInput = 4
    LoadMicroBatchTarget = 5
    OptimizerStep = 6
    RecvActivations = 7
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
        This returns a generator, which contains all the operations to
        process a single batch
        """
        pass

    @property
    @abstractmethod
    def num_buffers(self):
        """
        The number of buffers necessary for sending & receiving data.
        This should always be a multiple of 2, since we have input buffers and
        corresponding output buffers (at least during training)
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
        yield [(PipeInstr.ZeroGrad, {})]

        for mubatch_id in range(self.num_micro_batches):
            cmds = []

            if self.is_first_stage:
                cmds.append(
                    (
                        PipeInstr.LoadMicroBatchInput,
                        {"mubatch_id": mubatch_id, "buffer_idx": 0},
                    )
                )
            else:
                cmds.append((PipeInstr.RecvActivations, {"buffer_idx": 0}))

            cmds.append((PipeInstr.Forward, {"buffer_idx": 0}))

            if self.is_last_stage:
                cmds.append(
                    (
                        PipeInstr.LoadMicroBatchTarget,
                        {"mubatch_id": mubatch_id, "buffer_idx": 0},
                    )
                )
            else:
                cmds.append((PipeInstr.SendActivations, {"buffer_idx": 0}))
                cmds.append((PipeInstr.ReceiveOutputGrad, {"buffer_idx": 0}))

            if mubatch_id == self.num_micro_batches - 1:
                cmds.append((PipeInstr.BackwardGradAllReduce, {"buffer_idx": 0}))
            else:
                cmds.append((PipeInstr.BackwardGradAccumulate, {"buffer_idx": 0}))

            if not self.is_first_stage:
                cmds.append((PipeInstr.SendInputGrad, {"buffer_idx": 0}))

            if mubatch_id == self.num_micro_batches - 1:
                cmds.append((PipeInstr.OptimizerStep, {}))
            yield cmds

    @property
    def num_buffers(self):
        # need 1 Buffer for receiving input and 1 buffer for sending output
        # since this is naive PP, there's only ever one μB in flight at the
        # same time
        return 2


class InferenceSchedule(Schedule):
    def steps(self):
        for mubatch_id in range(self.num_micro_batches):
            cmds = []

            if self.is_first_stage:
                cmds.append(
                    (
                        PipeInstr.LoadMicroBatchInput,
                        {"mubatch_id": mubatch_id, "buffer_idx": 0},
                    )
                )
            else:
                cmds.append((PipeInstr.RecvActivations, {"buffer_idx": 0}))

            cmds.append((PipeInstr.Forward, {"buffer_idx": 0}))

            if not self.is_last_stage:
                cmds.append((PipeInstr.SendActivations, {"buffer_idx": 0}))
            yield cmds

    @property
    def num_buffers(self):
        # Could be done with 1 buffer (by doing the FWD inplace)
        return 2


class GPipeSchedule:
    def __init__(self):
        pass


class PipeDreamSchedule:
    def __init__(self):
        pass


def backprop_allreduce_gradient(comm, param):
    """
    start a non-blocking AllReduce for the parameters for which we just
    calculated the final gradient.
    This interleaves communication of this layer's gradients with
    computation of the next layers gradients

    Starting a new communication for each parameter is quite wasteful, particularly if
    the parameters are small. PyTorch's DDP implementation uses bucketing to get around this.
    """
    if param.requires_grad:
        # we won't be touching param.grad until the Op is done, so we do it inplace
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
    The buffers don't keep any state between batches.
    """

    input_buffers = None
    output_buffers = None

    def __init__(
        self,
        dp_comm: MPI.Comm,
        pp_comm: MPI.Comm,
        model: MLP,
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

    def execute(self, sched, batch_id):
        """
        Setup buffers and use the configured schedule to execute a full batch

        Basically it'll just call the right function, given whatever the scheduler
        tells it to do
        """

        # The buffers hold activations during FWD passes and gradients during BWD
        # activation.shape == grad.shape, hence we can reuse the same buffers for FWD & BWD
        # TODO make buffers persistent for the whole training run by setting μBatch-size during
        #   validation loss calculation
        assert sched.num_buffers % 2 == 0
        self.input_buffers = [
            np.empty((self.dataset.mubatch_size, self.model.in_dim), dtype=np.float32)
            for _ in range(sched.num_buffers // 2)
        ]
        self.output_buffers = [
            np.empty((self.dataset.mubatch_size, self.model.out_dim), dtype=np.float32)
            for _ in range(sched.num_buffers // 2)
        ]

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
                    case PipeInstr.RecvActivations:
                        self.recv_activations(
                            from_idx=self.get_predecessor(sched), **kwargs
                        )
                    case PipeInstr.SendActivations:
                        self.send_activations(
                            to_idx=self.get_successor(sched), **kwargs
                        )
                    case PipeInstr.ReceiveOutputGrad:
                        self.recv_grad(from_idx=self.get_successor(sched), **kwargs)
                    case PipeInstr.SendInputGrad:
                        self.send_grad(to_idx=self.get_predecessor(sched), **kwargs)
                    case _:
                        raise NotImplementedError(command)

    def load_micro_batch_input(self, batch_id, mubatch_id, buffer_idx):
        data = self.dataset.load_micro_batch_input(batch_id, mubatch_id)
        assert data.shape == self.input_buffers[buffer_idx].shape
        self.input_buffers[buffer_idx] = data

    def load_micro_batch_target(self, batch_id, mubatch_id, buffer_idx):
        data = self.dataset.load_micro_batch_target(batch_id, mubatch_id)
        assert self.output_buffers[buffer_idx].shape == data.shape
        self.output_buffers[buffer_idx] = data

    def get_predecessor(self, schedule):
        pred = self.stage_id - 1
        assert schedule.is_valid_stage_id(pred)
        return pred

    def get_successor(self, schedule):
        succ = self.stage_id + 1
        assert schedule.is_valid_stage_id(succ)
        return succ

    def send_activations(self, to_idx, buffer_idx):
        # send forwards
        self.pp_comm.Send(self.output_buffers[buffer_idx], to_idx)

    def recv_activations(self, from_idx, buffer_idx):
        # receive from previous
        self.pp_comm.Recv(self.input_buffers[buffer_idx], from_idx)

    def send_grad(self, to_idx, buffer_idx):
        # send backwards
        self.pp_comm.Send(self.input_buffers[buffer_idx], to_idx)

    def recv_grad(self, from_idx, buffer_idx):
        # receive from next
        self.pp_comm.Recv(self.output_buffers[buffer_idx], from_idx)

    def forward(self, buffer_idx):
        # FWD pass transforms input buffer into output buffer
        self.output_buffers[buffer_idx] = self.model.forward(
            self.input_buffers[buffer_idx]
        )

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
        # BWD pass transforms output buffer into input buffer
        self.input_buffers[buffer_idx] = self.model.backward(
            self.output_buffers[buffer_idx]
        )

    def optimizer_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.model.zero_grad()
