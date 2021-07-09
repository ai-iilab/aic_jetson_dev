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
    
    def init_calibration(self, calibration_file):
        cameraMarix_left = cameraMatrix_right = map_left_y = map_left_x = map_right_y = map_right_x = np.array([])
        
        config = configparser.ConfigParser()
        config.read(calibration_file)
        
        check_data = True
        resolution_str = ''
        if self.width == 2208 :
            resolution_str = '2K'
        elif self.width == 1920 :
            resolution_str = 'FHD'
        elif self.width == 1280 :
            resolution_str = 'HD'
        elif self.width == 672 :
            resolution_str = 'VGA'
        else:
            resolution_str = 'HD'
            check_data = False
        
        T_ = np.array([-float(config['STEREO']['Baseline'] if 'Baseline' in config['STEREO'] else 0),
                       float(config['STEREO']['TY_'+resolution_str] if 'TY_'+resolution_str in config['STEREO'] else 0),
                       float(config['STEREO']['TZ_'+resolution_str] if 'TZ_'+resolution_str in config['STEREO'] else 0)])
        
        
        left_cam_cx = float(config['LEFT_CAM_'+resolution_str]['cx'] if 'cx' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_cy = float(config['LEFT_CAM_'+resolution_str]['cy'] if 'cy' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_fx = float(config['LEFT_CAM_'+resolution_str]['fx'] if 'fx' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_fy = float(config['LEFT_CAM_'+resolution_str]['fy'] if 'fy' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_k1 = float(config['LEFT_CAM_'+resolution_str]['k1'] if 'k1' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_k2 = float(config['LEFT_CAM_'+resolution_str]['k2'] if 'k2' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_p1 = float(config['LEFT_CAM_'+resolution_str]['p1'] if 'p1' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_p2 = float(config['LEFT_CAM_'+resolution_str]['p2'] if 'p2' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_p3 = float(config['LEFT_CAM_'+resolution_str]['p3'] if 'p3' in config['LEFT_CAM_'+resolution_str] else 0)
        left_cam_k3 = float(config['LEFT_CAM_'+resolution_str]['k3'] if 'k3' in config['LEFT_CAM_'+resolution_str] else 0)
        
        
        right_cam_cx = float(config['RIGHT_CAM_'+resolution_str]['cx'] if 'cx' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_cy = float(config['RIGHT_CAM_'+resolution_str]['cy'] if 'cy' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_fx = float(config['RIGHT_CAM_'+resolution_str]['fx'] if 'fx' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_fy = float(config['RIGHT_CAM_'+resolution_str]['fy'] if 'fy' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_k1 = float(config['RIGHT_CAM_'+resolution_str]['k1'] if 'k1' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_k2 = float(config['RIGHT_CAM_'+resolution_str]['k2'] if 'k2' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_p1 = float(config['RIGHT_CAM_'+resolution_str]['p1'] if 'p1' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_p2 = float(config['RIGHT_CAM_'+resolution_str]['p2'] if 'p2' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_p3 = float(config['RIGHT_CAM_'+resolution_str]['p3'] if 'p3' in config['RIGHT_CAM_'+resolution_str] else 0)
        right_cam_k3 = float(config['RIGHT_CAM_'+resolution_str]['k3'] if 'k3' in config['RIGHT_CAM_'+resolution_str] else 0)
        
        R_zed = np.array([float(config['STEREO']['RX_'+resolution_str] if 'RX_' + resolution_str in config['STEREO'] else 0),
                          float(config['STEREO']['CV_'+resolution_str] if 'CV_' + resolution_str in config['STEREO'] else 0),
                          float(config['STEREO']['RZ_'+resolution_str] if 'RZ_' + resolution_str in config['STEREO'] else 0)])
        
        R, _ = cv2.Rodrigues(R_zed)
        cameraMatrix_left = np.array([[left_cam_fx, 0, left_cam_cx],
                             [0, left_cam_fy, left_cam_cy],
                             [0, 0, 1]])
        
        cameraMatrix_right = np.array([[right_cam_fx, 0, right_cam_cx],
                              [0, right_cam_fy, right_cam_cy],
                              [0, 0, 1]])
        
        distCoeffs_left = np.array([[left_cam_k1], [left_cam_k2], [left_cam_p1], [left_cam_p2], [left_cam_k3]])
        
        distCoeffs_right = np.array([[right_cam_k1], [right_cam_k2], [right_cam_p1], [right_cam_p2], [right_cam_k3]])
        
        T = np.array([[T_[0]], [T_[1]], [T_[2]]])
        R1 = R2 = P1 = P2 = np.array([])
        
        R1, R2, P1, P2 = cv2.stereoRectify(cameraMatrix1=cameraMatrix_left,
                                           cameraMatrix2=cameraMatrix_right,
                                           distCoeffs1=distCoeffs_left,
                                           distCoeffs2=distCoeffs_right,
                                           R=R, T=T,
                                           flags=cv2.CALIB_ZERO_DISPARITY,
                                           alpha=0,
                                           imageSize=(self.width, self.height),
                                           newImageSize=(self.width, self.height))[0:4]
        
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, (self.width, self.height), cv2.CV_32FC1)
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, (self.width, self.height), cv2.CV_32FC1)
        
        cameraMatrix_left = P1
        cameraMatrix_right = P2
        
        return cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y
    
    def capture_left(self):
        retval, frame = self.cap.read()
        left_right_image = np.split(frame, 2, axis=1)
        left_rect = cv2.remap(left_right_image[0], self.map_left_x, self.map_left_y, interpolation=cv2.INTER_LINEAR)
        return left_rect
        
    def capture_right(self):
        retval, frame = self.cap.read()
        left_right_image = np.split(frame, 2, axis=1)
        right_rect = cv2.remap(left_right_image[1], self.map_right_x, self.map_right_y, interpolation=cv2.INTER_LINEAR)
        return right_rect
        
    def capture_stereo(self):
