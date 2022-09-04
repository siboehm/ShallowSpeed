import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

from minMLP.dataset import Dataset
from minMLP.layers import MLP


class PipeInstr:
    pass


@dataclass
class ZeroGrad(PipeInstr):
    pass


@dataclass
class OptimizerStep(PipeInstr):
    pass


@dataclass
class BufferPipeInstr(PipeInstr):
    buffer_id: int


@dataclass
class RecvActivations(BufferPipeInstr):
    pass


@dataclass
class SendActivations(BufferPipeInstr):
    pass


@dataclass
class RecvOutputGrad(BufferPipeInstr):
    pass


@dataclass
class SendInputGrad(BufferPipeInstr):
    pass


@dataclass
class MuBatchPipeInstr(PipeInstr):
    buffer_id: int
    mubatch_id: int


@dataclass
class Forward(MuBatchPipeInstr):
    pass


@dataclass
class BackwardGradAcc(MuBatchPipeInstr):
    pass


@dataclass
class BackwardGradAllReduce(MuBatchPipeInstr):
    pass


@dataclass
class LoadInstruction(MuBatchPipeInstr):
    pass


@dataclass
class LoadMuBatchInput(LoadInstruction):
    pass


@dataclass
class LoadMuBatchTarget(LoadInstruction):
    pass


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


class NaiveParallelSchedule(Schedule):
    """
    A pipeline schedule without any interleaving of μBatches.
    Only one pipeline stage is activate at any given time
    """

    def steps(self):
        yield [ZeroGrad()]
        for mubatch_id in range(self.num_micro_batches):
            yield self.steps_mubatch(mubatch_id)
        # updating the weights is the last step of processing a batch
        yield [OptimizerStep()]

    def steps_mubatch(self, mubatch_id):
        cmds = []
        if self.is_first_stage:
            cmds.append(LoadMuBatchInput(mubatch_id=mubatch_id, buffer_id=0))
        else:
            cmds.append((RecvActivations(buffer_id=0)))
        cmds.append(Forward(buffer_id=0, mubatch_id=mubatch_id))
        if self.is_last_stage:
            cmds.append(LoadMuBatchTarget(mubatch_id=mubatch_id, buffer_id=0))
        else:
            cmds.append(SendActivations(buffer_id=0))
            cmds.append(RecvOutputGrad(buffer_id=0))
        if mubatch_id == self.num_micro_batches - 1:
            cmds.append(BackwardGradAllReduce(buffer_id=0, mubatch_id=mubatch_id))
        else:
            cmds.append(BackwardGradAcc(buffer_id=0, mubatch_id=mubatch_id))
        if not self.is_first_stage:
            cmds.append(SendInputGrad(buffer_id=0))
        return cmds

    @property
    def num_buffers(self):
        # need 1 Buffer for receiving input and 1 buffer for sending output
        # since this is naive PP, there's only ever one μB in flight at the
        # same time
        return 2


class GPipeSchedule(Schedule):
    def steps(self):
        yield [ZeroGrad()]

        # STAGE 1: FWD all μBatches
        for mubatch_id in range(self.num_micro_batches):
            yield self.steps_FWD_mubatch(mubatch_id)

        # STAGE 2: BWD all μBatches
        for mubatch_id in reversed(range(self.num_micro_batches)):
            yield from self.steps_BWD_mubatch(mubatch_id)

        # updating the weights is the last step of processing any batch
        yield [OptimizerStep()]

    def steps_BWD_mubatch(self, mubatch_id):
        cmds = []
        if self.is_last_stage:
            cmds.append(LoadMuBatchTarget(mubatch_id=mubatch_id, buffer_id=0))
        else:
            cmds.append(RecvOutputGrad(buffer_id=0))
        if mubatch_id == 0:
            # interleaved backprop & AllReduce during last μBatch
            cmds.append(BackwardGradAllReduce(buffer_id=0, mubatch_id=mubatch_id))
        else:
            cmds.append(BackwardGradAcc(buffer_id=0, mubatch_id=mubatch_id))
        if not self.is_first_stage:
            cmds.append(SendInputGrad(buffer_id=0))
        yield cmds

    def steps_FWD_mubatch(self, mubatch_id):
        cmds = []
        if self.is_first_stage:
            cmds.append(LoadMuBatchInput(buffer_id=0, mubatch_id=mubatch_id))
        else:
            cmds.append(RecvActivations(buffer_id=0))
        cmds.append(Forward(buffer_id=0, mubatch_id=mubatch_id))
        # the last stage just discards the output of its `forward()` pass since
        # it's not necessary for running BWD. The last stage just needs the target values
        # (loaded from disk) and the activations (cached inside the `Module`'s) for BWD.
        if not self.is_last_stage:
            cmds.append(SendActivations(buffer_id=0))
        return cmds

    @property
    def num_buffers(self):
        # could keep more buffers around and make the sending & receiving async
        return 2


class InferenceSchedule(Schedule):
    def steps(self):
        for mubatch_id in range(self.num_micro_batches):
            cmds = []

            if self.is_first_stage:
                cmds.append(LoadMuBatchInput(mubatch_id=mubatch_id, buffer_id=0))
            else:
                cmds.append(RecvActivations(buffer_id=0))

            cmds.append(Forward(buffer_id=0, mubatch_id=mubatch_id))

            if not self.is_last_stage:
                cmds.append(SendActivations(buffer_id=0))
            yield cmds

    @property
    def num_buffers(self):
        # Could be done with 1 buffer (by doing the FWD inplace)
        return 2


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

    def load_micro_batch_input(self, batch_id, mubatch_id, buffer_id):
        data = self.dataset.load_micro_batch_input(batch_id, mubatch_id)
        assert (
            data.shape == self.input_buffers[buffer_id].shape
        ), f"shape is {data.shape} but should be {self.input_buffers[buffer_id].shape}"
        self.input_buffers[buffer_id] = data

    def load_micro_batch_target(self, batch_id, mubatch_id, buffer_id):
        data = self.dataset.load_micro_batch_target(batch_id, mubatch_id)
        assert self.output_buffers[buffer_id].shape == data.shape
        self.output_buffers[buffer_id] = data

    def send_activations(self, buffer_id):
        # send forwards
        self.pp_comm.Send(self.output_buffers[buffer_id], self.get_successor())

    def recv_activations(self, buffer_id):
        # receive from previous
        self.pp_comm.Recv(self.input_buffers[buffer_id], self.get_predecessor())

    def send_grad(self, buffer_id):
        # send backwards
        self.pp_comm.Send(self.input_buffers[buffer_id], self.get_predecessor())

    def recv_grad(self, buffer_id):
        # receive from next
        self.pp_comm.Recv(self.output_buffers[buffer_id], self.get_successor())

    def forward(self, buffer_id, mubatch_id):
        # FWD pass transforms input buffer into output buffer
        self.output_buffers[buffer_id] = self.model.forward(
            inputs=self.input_buffers[buffer_id], mubatch_id=mubatch_id
        )

    def backward_and_reduce(self, buffer_id, mubatch_id):
        # hooks for AllReducing-ing the gradient across all dp_workers
        self.model.register_grad_hook(
            lambda param: backprop_allreduce_gradient(self.dp_comm, param)
        )
        self.model.register_post_grad_hook(backprop_block_for_comms)

        # regular backwards pass will trigger the AR hooks
        self.backward_accumulate(buffer_id, mubatch_id=mubatch_id)

        self.model.reset_grad_hooks()
        self.model.reset_post_grad_hooks()

    def backward_accumulate(self, buffer_id, mubatch_id):
        # BWD pass transforms output buffer into input buffer
        self.input_buffers[buffer_id] = self.model.backward(
            dout=self.output_buffers[buffer_id], mubatch_id=mubatch_id
        )

    def optimizer_step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.model.zero_grad()

    def get_predecessor(self):
        return self.stage_id - 1

    def get_successor(self):
        return self.stage_id + 1

    _INSTRUCTION_MAP = {
        LoadMuBatchInput: load_micro_batch_input,
        LoadMuBatchTarget: load_micro_batch_target,
        Forward: forward,
        BackwardGradAllReduce: backward_and_reduce,
        BackwardGradAcc: backward_accumulate,
        OptimizerStep: optimizer_step,
        ZeroGrad: zero_grad,
        RecvActivations: recv_activations,
        SendActivations: send_activations,
        RecvOutputGrad: recv_grad,
        SendInputGrad: send_grad,
    }

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
            for command in commands:
                if isinstance(command, LoadInstruction):
                    self._INSTRUCTION_MAP[type(command)](
                        self, batch_id, **dataclasses.asdict(command)
                    )
                else:
                    self._INSTRUCTION_MAP[type(command)](
                        self, **dataclasses.asdict(command)
                    )
