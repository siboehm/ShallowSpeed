from minMLP.pipe import DataParallelSchedule, PipelineInstruction


def test_dp_schedule():
    sched = DataParallelSchedule(num_micro_batches=5, num_stages=1, stage_id=0)
    cmds = list(sched.steps())
    assert PipelineInstruction.ZeroGrad in cmds[0]
    assert PipelineInstruction.BackwardGradReduce in cmds[-1]
    assert PipelineInstruction.BackwardGradReduce not in set().union(*cmds[1:-1])
