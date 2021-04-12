"""
File: Main.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

from PreProcessor import PreProcessor
from InferenceTRT import InferenceTRT
from PostProcessor import PostProcessor

ENABLE_TIME_PROFILE = False
CONF_THRESH = 0.1
IOU_THRESHOLD = 0.4

if __name__ == "__main__":
    pre_process_wrapper = PreProcessor((1080, 1920, 3), (608, 608, 3), ENABLE_TIME_PROFILE)
    inference_trt_wrapper = InferenceTRT("yolov5s_FP16.engine", ENABLE_TIME_PROFILE)
    post_process_wrapper = PostProcessor((1080, 1920, 3), (608, 608, 3), CONF_THRESH, IOU_THRESHOLD, ENABLE_TIME_PROFILE)

    pre_process_wrapper.destroy()
    inference_trt_wrapper.destroy()
    post_process_wrapper.destroy()
