from typing import List, Dict, Tuple

def dispatch_inputs(
    batch: Dict,
    align_fuse: List[List[str], List[str]],
    epoch: int
    ) -> Tuple[dict, dict, bool]:
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
    """
        
    if align_fuse[0] == align_fuse[1]:
        apply_mask = True
        student_index = 0
        teacher_index = 1
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
    return student_inputs, teacher_inputs, apply_mask
