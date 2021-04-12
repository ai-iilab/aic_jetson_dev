"""
File: Main.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import cv2

from PreProcessor import PreProcessor
from InferenceTRT import InferenceTRT
from PostProcessor import PostProcessor

ENABLE_WRITE_OUTPUT = True
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

class myThread(threading.Thread):
    def __init__(self, pre_proc, infer_proc, post_proc, img):
        threading.Thread.__init__(self)
        self.pre_proc = pre_proc
        self.infer_proc = infer_proc
        self.post_proc = post_proc
        self.img = img

    def run(self):
        self.pre_proc.preprocess_image(h_img, self.infer_proc.get_infer_ptr())
        infer_result = self.infer_proc.inference()
        result_boxes, result_scores, result_classid = self.post_proc.post_process(infer_result)
  
        for i in range(len(result_boxes)):
            box = result_boxes[i]
            plot_one_box(
                box,
                h_img,
                label="{}:{:.2f}".format(
                    categories[int(result_classid[i])], result_scores[i]
                ),
            )

        if ENABLE_WRITE_OUTPUT == True:
            save_name = "output/0000_V0000_%03d.jpg" % index
            cv2.imwrite(save_name, h_img)

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
        
        thread1 = myThread(pre_process_wrapper, inference_trt_wrapper, post_process_wrapper, h_img)
        thread1.start()
        thread1.join()
    
    
    if ENABLE_TIME_PROFILE == True:
        print("\n")
        print("Pre-process time  : ", pre_process_wrapper.proc_time / (end_frame - start_frame), " msec")
        print("Inference time    : ", inference_trt_wrapper.proc_time / (end_frame - start_frame), " msec")
        print("Post-process time : ", post_process_wrapper.proc_time / (end_frame - start_frame), " msec", "\n")
    
    pre_process_wrapper.destroy()
    inference_trt_wrapper.destroy()
    post_process_wrapper.destroy()
