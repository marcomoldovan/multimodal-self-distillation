from typing import List, Dict, Tuple

from src.models.components.outputs import DispatcherOutput

def dispatch_inputs(
    batch: Dict,
    epoch: int,
    switch_teacher_student: bool = False
    ) -> DispatcherOutput:
    """
    Returns the input dicts for student and teacher model.
    
    Parameters
    ----------
        batch : (dict)
            batch of data, must contain the key 'align_fuse' which specifies
            the alignment and fusion procedure and what to feed to the student 
            and teacher, examples:
            - [['text'], ['video', 'audio']]
            - [['video', 'audio'], ['video', audio']]
            - [['text'], ['audio']]
            - [['image'], ['image']]
        epoch : (int)
            number of current epoch
        switch_teacher_student : (bool)
        validation : (bool)
        test : (bool)
            
    Returns
    -------
        dispatcher_output : DispatcherOutput
    """
        
    if 'align_fuse' in batch.keys():
        align_fuse = batch['align_fuse']
    else:
        # for inference when only the input data is given and nothing else
        align_fuse = [[key] for key in batch.keys()]
    
    if 'labels' in batch.keys():
        labels = batch['labels']
    else:
        labels = None
        
    if 'metric' in batch.keys():
        metric = batch['metric']
    else:
        metric = None
        
    if 'num_classes' in batch.keys():
        num_classes = batch['num_classes']
    else:
        num_classes = None
        
    if align_fuse[0] == align_fuse[1]:
        # unimodal case, e.g. [['text'], ['text']] or [['image'], ['image']]
        apply_mask = True
        student_index = 0
        teacher_index = 1
    elif len(align_fuse) == 1: 
        # inference is assumed here with align_fuse like [['image']] or [['video', 'audio']]
        apply_mask = False
        student_index = 0
        teacher_index = 0
    else:
        # multimodal case, e.g. [['text'], ['video', 'audio']] or [['text'], ['audio']]
        apply_mask = False
        if switch_teacher_student:
            if epoch % 2 == 0:
                student_index = 0
                teacher_index = 1
            else:
                student_index = 1
                teacher_index = 0
        else:
            student_index = 0
            teacher_index = 1
        
    student_inputs = {}
    teacher_inputs = {}
    
    for k, v in batch.items():
        if k in align_fuse[student_index]:
            student_inputs[k] = v
        elif k in align_fuse[teacher_index]:
            teacher_inputs[k] = v
            
    output_modalities = {'student_output': student_inputs.keys(), 'teacher_output': teacher_inputs.keys()}
    
    dispater_output = DispatcherOutput(
        student_input=student_inputs,
        teacher_inputs=teacher_inputs,
        align_fuse=align_fuse,
        apply_mask=apply_mask,
        labels=labels,
        output_modalities=output_modalities,
        metric=metric,
        num_classes=num_classes,
    )
    
    return dispater_output