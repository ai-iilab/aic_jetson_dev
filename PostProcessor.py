"""
File: PostProcessor.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

class PostProcessor(object):
    """
    description: A PostProcessor class that warps postprocess ops.
    """
    
    def __init__(self, conf_threshold, iou_threshold):
        print("PostProcessor init")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def destroy(self):
        print("PostProcessor destroy")
