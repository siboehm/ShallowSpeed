from minMLP.pipe import DataParallelSchedule, PipelineInstruction


def test_dp_schedule():
    sched = DataParallelSchedule(num_micro_batches=5, num_stages=1, stage_id=0)
    cmds = [[cmd for cmd, kwargs in cmds] for cmds in sched.steps()]
    assert PipelineInstruction.ZeroGrad in cmds[0]

    # for the final microbatch we AllReduce and step
    assert PipelineInstruction.BackwardGradAllReduce in cmds[-1]
    assert PipelineInstruction.OptimizerStep in cmds[-1]
    assert PipelineInstruction.BackwardGradAllReduce not in set().union(*cmds[:-1])
    assert PipelineInstruction.OptimizerStep not in set().union(*cmds[:-1])
