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
    
    def __init__(self):
        print("CameraZED init")
    
    def destroy(self):
        print("CameraZED destroy")