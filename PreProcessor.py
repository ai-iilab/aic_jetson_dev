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
    
    def __init__(self):
        print("PreProcessor init")
        
        cfx = cuda.Device(0).make_context()
        
        self.cfx = cfx
        
    def destroy(self):
        print("PreProcessor destroy")
        
        self.cfx.pop()
        
    def preprocess_image(self):
        """
        description: 
        param: 
        return: 
        """
