"""
File: PreProcessor.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import pycuda.autoinit
import pycuda.driver as cuda

class PreProcessor(object):
    """
    description: A PreProcessor class that warps preprocess ops.
    """
    
    def __init__(self, input_shape, infer_shape):
        print("PreProcessor init")
        
        cfx = cuda.Device(0).make_context()
        
        input_height, input_width, input_channel = input_shape
        infer_height, infer_width, infer_channel = infer_shape
        
        self.cfx = cfx
        
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        
        self.infer_width = infer_width
        self.infer_height = infer_height
        self.infer_channel = infer_channel
        
    def destroy(self):
        print("PreProcessor destroy")
        
        self.cfx.pop()
        
    def preprocess_image(self):
        """
        description: 
        param: 
        return: 
        """
