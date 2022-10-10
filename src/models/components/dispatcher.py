
def dispatch_inputs(
    batch: dict,
    fuse: list[str],
    align: list[str]
    ):
    """Dispatch modalities to their respective encoders."""
    student_inputs = {}
    teacher_inputs = {}
    for k, v in batch.items():
        if k in fuse:
            teacher_inputs[k] = v
    return student_inputs, teacher_inputs