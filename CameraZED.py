"""
File: CameraZED.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import sys
import os
import numpy as np
import configparser
import cv2
import wget

class CameraZED(object):
    """
    description: A CameraZED class that warps image capture ops.
    """
    
    def __init__(self, serial_number, image_width, image_height, fps):
        print("CameraZED init")
        
        self.serial_number = serial_number
        self.width = image_width
        self.height = image_height
        self.fps = fps
        
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened() == 0:
            exit(-1)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width * 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def destroy(self):
        print("CameraZED destroy")