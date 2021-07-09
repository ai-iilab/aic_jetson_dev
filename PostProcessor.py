"""
File: PostProcessor.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch
import torchvision

class PostProcessor(object):
    """
    description: A PostProcessor class that warps postprocess ops.
    """
    
    def __init__(self, input_shape, infer_shape, conf_threshold, iou_threshold, enable_profiling):
        print("PostProcessor init")
        
        max_batch_size, input_height, input_width, input_channel = input_shape
        max_batch_size, infer_height, infer_width, infer_channel = infer_shape
        
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        
        self.infer_width = infer_width
        self.infer_height = infer_height
        self.infer_channel = infer_channel
        
        self.max_batch_size = max_batch_size
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.proc_time = 0
        self.enable_profiling = enable_profiling
    
    def destroy(self):
        print("PostProcessor destroy")

    def xywh2xyxy(self, infer_h, infer_w, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            infer_h:    height of inference image
            infer_w:    width of inference image
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = infer_w / origin_w
        r_h = infer_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (infer_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (infer_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (infer_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (infer_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y
    
    def post_process(self, output):
        """
        description: postprocess the prediction
        param:
            output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        
        if self.enable_profiling == True:
            start = time.time()
        
        input_width = self.input_width
        input_height = self.input_height
        
        infer_width = self.infer_width
        infer_height = self.infer_height
        
        max_batch_size = self.max_batch_size
        
        conf_threshold = self.conf_threshold
        iou_threshold = self.iou_threshold
        
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        scores = pred[:, 4]
        si = scores > conf_threshold
        pred = pred[si, :]
        
        pred = torch.Tensor(pred).cpu()
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classid = pred[:, 5]
        
        #boxes = boxes[si, :]
        #scores = scores[si]
        #classid = classid[si]
        
        boxes = self.xywh2xyxy(infer_height, infer_width, input_height, input_width, boxes)
        
        indices = torchvision.ops.nms(boxes, scores, iou_threshold=iou_threshold).cpu()
        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()
        
        if self.enable_profiling == True:
            end = time.time()
            self.proc_time += (end - start) * 1000
        
        return result_boxes, result_scores, result_classid
