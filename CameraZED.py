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
        
        calibration_file = self.download_calibration_file(self.serial_number)
        if calibration_file  == "":
            exit(1)
        
        print("Calibration file found. Loading...")
        camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y = self.init_calibration(calibration_file)
        
        self.camera_matrix_left = camera_matrix_left
        self.camera_matrix_right = camera_matrix_right
        self.map_left_x = map_left_x
        self.map_left_y = map_left_y
        self.map_right_x = map_right_x
        self.map_right_y = map_right_y
    
    def destroy(self):
        print("CameraZED destroy")
        
        self.cap.release()
    
    def download_calibration_file(self, serial_number) :
        if os.name == 'nt' :
            #hidden_path = os.getenv('APPDATA') + '\\Stereolabs\\settings\\'
            hidden_path = 'C:\\ProgramData' + '\\Stereolabs\\settings\\'
        else :
            hidden_path = '/usr/local/zed/settings/'
        calibration_file = hidden_path + 'SN' + str(serial_number) + '.conf'
        
        if os.path.isfile(calibration_file) == False:
            url = 'http://calib.stereolabs.com/?SN='
            filename = wget.download(url=url+str(serial_number), out=calibration_file)
        
            if os.path.isfile(calibration_file) == False:
                print('Invalid Calibration File')
                return ""
        
        return calibration_file
