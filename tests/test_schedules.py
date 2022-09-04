from minMLP.pipe import DataParallelSchedule, PipeInstr


def _sched_to_list(sched):
    return [[(cmd, kwargs) for cmd, kwargs in cmds] for cmds in sched.steps()]


def test_dp_only_schedule():
    sched = DataParallelSchedule(num_micro_batches=5, num_stages=1, stage_id=0)
    cmds = [[cmd for cmd, kwargs in cmds] for cmds in sched.steps()]
    assert PipeInstr.ZeroGrad in cmds[0]

    # for the final microbatch we AllReduce and step
    assert PipeInstr.BackwardGradAllReduce in cmds[-1]
    assert PipeInstr.OptimizerStep in cmds[-1]
    assert PipeInstr.BackwardGradAllReduce not in set().union(*cmds[:-1])
    assert PipeInstr.OptimizerStep not in set().union(*cmds[:-1])


def test_pp_only_schedule():
    first_sched = DataParallelSchedule(num_micro_batches=1, num_stages=2, stage_id=0)
    first_cmds = _sched_to_list(first_sched)
    # assert PipeInstr.ZeroGrad in first_cmds[0]

    second_sched = DataParallelSchedule(num_micro_batches=1, num_stages=2, stage_id=1)
    second_cmds = _sched_to_list(second_sched)
    # assert PipeInstr.ZeroGrad in second_cmds[0]

    first_sched = DataParallelSchedule(num_micro_batches=1, num_stages=2, stage_id=0)
    cmds = [[(cmd, kwargs) for cmd, kwargs in cmds] for cmds in first_sched.steps()]
    assert PipeInstr.ZeroGrad in cmds[0]

    # for the final microbatch we AllReduce and step
    assert PipeInstr.BackwardGradAllReduce in cmds[-1]
    assert PipeInstr.OptimizerStep in cmds[-1]
    assert PipeInstr.BackwardGradAllReduce not in set().union(*cmds[:-1])
    assert PipeInstr.OptimizerStep not in set().union(*cmds[:-1])
