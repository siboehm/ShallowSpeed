import shallowspeed.pipe as pipe
from shallowspeed.pipe import GPipeSchedule, NaiveParallelSchedule, PipeInstr

"""
TODO these tests are fairly useless. 
How to improve:
- define a "happens before" predicate and eg test:
    - happens_before(BWD_MuBatch1, FWD_MuBatch2) in Naive schedule
    - happens_before(FWD_MuBatch<last>, BWD_MuBatch1) in GPipe
"""


def flatten(sched):
    if not isinstance(sched[0], list):
        return sched

    flat_sched = []
    for cmds in sched:
        for cmd in cmds:
            flat_sched.append(cmd)
    return flat_sched


def cmd_is_in(cmd_t, sched):
    flat_sched = flatten(sched)
    return any(isinstance(x, cmd_t) for x in flat_sched)


def test_naive_pp_schedule_dp_only():
    # naive scheduler without any pipeline parallelism (num_stages=1)
    sched = NaiveParallelSchedule(num_micro_batches=5, num_stages=1, stage_id=0)
    cmds = list(sched.steps())
    assert cmd_is_in(pipe.ZeroGrad, cmds[0])
    assert not cmd_is_in(pipe.ZeroGrad, cmds[1:])

    # for the final microbatch we AllReduce and step
    assert cmd_is_in(pipe.BackwardGradAllReduce, cmds[-2])
    assert cmd_is_in(pipe.OptimizerStep, cmds[-1])
    assert not cmd_is_in(pipe.BackwardGradAllReduce, cmds[:-2])
    assert not cmd_is_in(pipe.OptimizerStep, cmds[:-1])


def test_naive_pp_schedule_pp_only():
    # naive scheduler without any μBatches parallelism
    first_sched = NaiveParallelSchedule(num_micro_batches=1, num_stages=2, stage_id=0)
    first_cmds = list(first_sched.steps())
    # init
    assert cmd_is_in(pipe.ZeroGrad, first_cmds[0])
    assert cmd_is_in(pipe.LoadMuBatchInput, first_cmds[1])
    assert not cmd_is_in(pipe.LoadMuBatchTarget, first_cmds)
    # finish
    assert cmd_is_in(pipe.BackwardGradAllReduce, first_cmds[-2])
    assert not cmd_is_in(pipe.BackwardGradAllReduce, first_cmds[:-2])
    assert cmd_is_in(pipe.OptimizerStep, first_cmds[-1])
    assert not cmd_is_in(pipe.OptimizerStep, first_cmds[:-1])

    second_sched = NaiveParallelSchedule(num_micro_batches=1, num_stages=2, stage_id=1)
    second_cmds = list(second_sched.steps())
    # init
    assert cmd_is_in(pipe.ZeroGrad, second_cmds[0])
    assert cmd_is_in(pipe.RecvActivations, second_cmds[1])
    assert not cmd_is_in(pipe.LoadMuBatchInput, second_cmds)
    # processing
    assert cmd_is_in(pipe.LoadMuBatchTarget, second_cmds[1])
    # finish
    assert cmd_is_in(pipe.BackwardGradAllReduce, second_cmds[-2])
    assert not cmd_is_in(pipe.BackwardGradAllReduce, second_cmds[:-2])
    assert cmd_is_in(pipe.OptimizerStep, first_cmds[-1])
    assert not cmd_is_in(pipe.OptimizerStep, second_cmds[:-1])


def test_gpipe_schedule():
    first_sched = GPipeSchedule(num_micro_batches=2, num_stages=3, stage_id=0)
    first_cmds = list(first_sched.steps())

    # init
    assert cmd_is_in(pipe.ZeroGrad, first_cmds[0])
    assert cmd_is_in(pipe.LoadMuBatchInput, first_cmds[1])
    assert not cmd_is_in(pipe.LoadMuBatchTarget, first_cmds)
    # finish
    assert cmd_is_in(pipe.BackwardGradAllReduce, first_cmds[-2])
    assert not cmd_is_in(pipe.BackwardGradAllReduce, first_cmds[:-2])
    assert cmd_is_in(pipe.OptimizerStep, first_cmds[-1])
    assert not cmd_is_in(pipe.OptimizerStep, first_cmds[:-1])

    second_sched = GPipeSchedule(num_micro_batches=2, num_stages=3, stage_id=1)
    second_cmds = list(second_sched.steps())
    # init
    assert cmd_is_in(pipe.ZeroGrad, second_cmds[0])
    assert cmd_is_in(pipe.RecvActivations, second_cmds[1])
    assert not cmd_is_in(pipe.LoadMuBatchInput, second_cmds)
    assert not cmd_is_in(pipe.LoadMuBatchTarget, second_cmds)
    # processing
    assert cmd_is_in(pipe.SendActivations, second_cmds[1:])
    assert cmd_is_in(pipe.RecvActivations, second_cmds[1:])
    assert cmd_is_in(pipe.SendInputGrad, second_cmds[1:])
    assert cmd_is_in(pipe.RecvOutputGrad, second_cmds[1:])
    # finish
    assert cmd_is_in(pipe.BackwardGradAllReduce, second_cmds[-2])
    assert not cmd_is_in(pipe.BackwardGradAllReduce, second_cmds[:-2])
    assert cmd_is_in(pipe.OptimizerStep, first_cmds[-1])
    assert not cmd_is_in(pipe.OptimizerStep, second_cmds[:-1])
