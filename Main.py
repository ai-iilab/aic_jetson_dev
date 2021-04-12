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

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=2)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

if __name__ == "__main__":
    pre_process_wrapper = PreProcessor((1080, 1920, 3), (608, 608, 3), ENABLE_TIME_PROFILE)
    inference_trt_wrapper = InferenceTRT("yolov5s_FP16.engine", ENABLE_TIME_PROFILE)
    post_process_wrapper = PostProcessor((1080, 1920, 3), (608, 608, 3), CONF_THRESH, IOU_THRESHOLD, ENABLE_TIME_PROFILE)
    
    start_frame = 0
    end_frame = 1000
    for index in range(start_frame, end_frame):
        image_path = "image/0000_V0000_%03d.jpg" % index
        print(image_path)
        
        h_img = cv2.imread(image_path)
    
    pre_process_wrapper.destroy()
    inference_trt_wrapper.destroy()
    post_process_wrapper.destroy()
