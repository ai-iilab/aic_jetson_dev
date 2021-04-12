"""
File: PreProcessor.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import numpy
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

class PreProcessor(object):
    """
    description: A PreProcessor class that warps preprocess ops.
    """
    
    def __init__(self, input_shape, infer_shape):
        print("PreProcessor init")
        
        cfx = cuda.Device(0).make_context()
        
        input_height, input_width, input_channel = input_shape
        infer_height, infer_width, infer_channel = infer_shape
        
        d_img = gpuarray.empty(input_shape, numpy.uint8)
        d_img_temp = gpuarray.empty(infer_shape, numpy.uint8)
        d_img_resize = gpuarray.empty(infer_shape, numpy.uint8)
        
        pre_process_lib = ctypes.cdll.LoadLibrary("CUDA_PreProcessor.so")
        pre_process_lib.ImagePreProcessing.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_float), ctypes.c_int)
        
        self.cfx = cfx
        
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        
        self.infer_width = infer_width
        self.infer_height = infer_height
        self.infer_channel = infer_channel
        
        self.d_img = d_img
        self.d_img_temp = d_img_temp
        self.d_img_resize = d_img_resize
        
        self.pre_process_lib = pre_process_lib
        
    def destroy(self):
        print("PreProcessor destroy")
        
        self.cfx.pop()
        
    def preprocess_image(self, input_image, infer_ptr):
        """
        description: 
        param: 
        return: 
        """
        
        UCHARP = ctypes.POINTER(ctypes.c_ubyte)
        FLOATP = ctypes.POINTER(ctypes.c_float)
        VOIDP = ctypes.POINTER(ctypes.c_void_p)
        
        self.cfx.push()
        cuda.memcpy_htod(self.d_img.ptr, input_image.ravel())
        self.pre_process_lib.ImagePreProcessing(self.input_width, self.input_height, self.infer_width, self.infer_height, ctypes.cast(self.d_img.ptr, UCHARP), ctypes.cast(self.d_img_temp.ptr, UCHARP), ctypes.cast(self.d_img_resize.ptr, UCHARP), ctypes.cast(infer_ptr, FLOATP), 0)
        self.cfx.pop()
        
        return self.d_img_resize.ptr
