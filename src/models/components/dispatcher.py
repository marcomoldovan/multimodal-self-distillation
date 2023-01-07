from typing import List, Dict, Tuple

def dispatch_inputs(
    batch: Dict,
    epoch: int
    ) -> Tuple[dict, dict, bool, dict]:
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
            
    Returns
    -------
        student_inputs : (dict)
        teacher_inputs : (dict)
        apply_align : (bool)
        labels : (dict)
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
        
    if align_fuse[0] == align_fuse[1]:
        apply_mask = True
        student_index = 0
        teacher_index = 1
    elif len(align_fuse) == 1: 
        # inference is assumed here with align_fuse like [['image']] or [['video', 'audio']]
        apply_mask = False
        student_index = 0
        teacher_index = 0
    else:
        apply_mask = False
        if epoch % 2 == 0:
            student_index = 0
            teacher_index = 1
        else:
            student_index = 1
            teacher_index = 0
        
    student_inputs = {}
    teacher_inputs = {}
    
    for k, v in batch.items():
        if k in align_fuse[student_index]:
            student_inputs[k] = v
        elif k in align_fuse[teacher_index]:
            teacher_inputs[k] = v
            
    output_modalities = {'student_output': student_inputs.keys(), 'teacher_output': teacher_inputs.keys()}
    
    return student_inputs, teacher_inputs, align_fuse, apply_mask, labels, output_modalities, metric