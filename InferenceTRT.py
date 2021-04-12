"""
File: InferenceTRT.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import time
import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

class InferenceTRT(object):
    """
    description: A InferenceTRT class that warps TensorRT ops
    """
    
    def __init__(self, engine_file_path, enable_profiling):
        print("InferenceTRT init")
        
        self.cfx = cuda.Device(0).make_context()
        ctypes.CDLL("libyolov5trt.so")
        
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        
        input_size = 0
        cuda_inputs = []
        cuda_outputs = []
        host_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            
            gpu_array = gpuarray.empty(size, dtype)
            cuda_mem = gpu_array.gpudata
            bindings.append(int(cuda_mem))
            
            if engine.binding_is_input(binding):
                cuda_inputs.append(cuda_mem)
                input_ptr = gpu_array.ptr
            else:
                cuda_outputs.append(cuda_mem)
                host_mem = cuda.pagelocked_empty(size, dtype)
                host_outputs.append(host_mem)
        
        self.context = context
        self.input_ptr = input_ptr
        self.cuda_inputs = cuda_inputs
        self.cuda_outputs = cuda_outputs
        self.host_outputs = host_outputs
        self.bindings = bindings
        
        self.proc_time = 0
        self.enable_profiling = enable_profiling
        
    def destroy(self):
        print("InferenceTRT destroy")
        
        self.cfx.pop()
        
    def inference(self):
        """
        description: Execute inference and return inference result
        param:
            None
        return:
            host_outputs: Inference result
        """
        
        if self.enable_profiling == True:
            start = time.time()
        
        context = self.context
        cuda_outputs = self.cuda_outputs
        host_outputs = self.host_outputs
        bindings = self.bindings
        
        self.cfx.push()
        context.execute(bindings=bindings)
        cuda.memcpy_dtoh(host_outputs[0], cuda_outputs[0])
        self.cfx.pop()
        
        if self.enable_profiling == True:
            end = time.time()
            self.proc_time += (end - start) * 1000
        
        return host_outputs[0]
        
    def get_infer_ptr(self):
        return self.input_ptr
