"""
File: InferenceTRT.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import ctypes
import pycuda.autoinit
import pycuda.driver as cuda

class InferenceTRT(object):
    """
    description: A InferenceTRT class that warps TensorRT ops
    """
    
    def __init__(self):
        print("InferenceTRT init")
        
        self.cfx = cuda.Device(0).make_context()
        ctypes.CDLL("libyolov5trt.so")
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        
        self.context = context
        
    def destroy(self):
        print("InferenceTRT destroy")
        
        self.cfx.pop()
