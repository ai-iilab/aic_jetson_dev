"""
File: Main.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""
import json
import random
import threading
import cv2

from CameraZED import CameraZED
from PreProcessor import PreProcessor
from InferenceTRT import InferenceTRT
from PostProcessor import PostProcessor

ENABLE_DUMMY_INPUT = True
ENABLE_DRAW_BOX = True
ENABLE_DRAW_FPS = False
ENABLE_TIME_PROFILE = True
ENABLE_CAMERA_LIVE = False
ENABLE_WRITE_JSON = False
ENABLE_SHOW_OUTPUT = False
ENABLE_WRITE_OUTPUT = True

INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080
INPUT_FPS = 30

INFER_WIDTH = 640
INFER_HEIGHT = 640

CONF_THRESH = 0.1
IOU_THRESHOLD = 0.4

BATCH_SIZE = 1
if BATCH_SIZE > 1:
    ENABLE_DRAW_FPS = False
    ENABLE_CAMERA_LIVE = False
    ENABLE_WRITE_JSON = False
    ENABLE_SHOW_OUTPUT = False

CAMERA_TOTAL_FRAME = 1000

ENGINE_PATH = "./yolov5s_w0.04_d0.33_BATCH32.engine"
JSON_PATH = "./result.json"

categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

text_color = [
    (0, 0, 0),     (63, 0, 0),     (127, 0, 0),     (191, 0, 0),     (255, 0, 0),
    (0, 0, 63),	   (63, 0, 63),    (127, 0, 63),    (191, 0, 63),    (255, 0, 63),
    (0, 0, 127),   (63, 0, 127),   (127, 0, 127),   (191, 0, 127),   (255, 0, 127),
    (0, 0, 191),   (63, 0, 191),   (127, 0, 191),   (191, 0, 191),   (255, 0, 191),
    (0, 0, 255),   (63, 0, 255),   (127, 0, 255),   (191, 0, 255),   (255, 0, 255),
    (0, 63, 0),    (63, 63, 0),    (127, 63, 0),    (191, 63, 0),    (255, 63, 0),
    (0, 63, 63),   (63, 63, 63),   (127, 63, 63),   (191, 63, 63),   (255, 63, 63),
    (0, 63, 127),  (63, 63, 127),  (127, 63, 127),  (191, 63, 127),  (255, 63, 127),
    (0, 63, 191),  (63, 63, 191),  (127, 63, 191),  (191, 63, 191),  (255, 63, 191),
    (0, 63, 255),  (63, 63, 255),  (127, 63, 255),  (191, 63, 255),  (255, 63, 255),
    (0, 127, 0),   (63, 127, 0),   (127, 127, 0),   (191, 127, 0),   (255, 127, 0),
    (0, 127, 63),  (63, 127, 63),  (127, 127, 63),  (191, 127, 63),  (255, 127, 63),
    (0, 127, 127), (63, 127, 127), (127, 127, 127), (191, 127, 127), (255, 127, 127),
    (0, 127, 191), (63, 127, 191), (127, 127, 191), (191, 127, 191), (255, 127, 191),
    (0, 127, 255), (63, 127, 255), (127, 127, 255), (191, 127, 255), (255, 127, 255),
    (0, 191, 0),   (63, 191, 0),   (127, 191, 0),   (191, 191, 0),   (255, 191, 0),
    (0, 191, 63),  (63, 191, 63),  (127, 191, 63),  (191, 191, 63),  (255, 191, 63),
    (0, 191, 127), (63, 191, 127), (127, 191, 127), (191, 191, 127), (255, 191, 127),
    (0, 191, 191), (63, 191, 191), (127, 191, 191), (191, 191, 191), (255, 191, 191),
    (0, 191, 255), (63, 191, 255), (127, 191, 255), (191, 191, 255), (255, 191, 255),
    (0, 255, 0),   (63, 255, 0),   (127, 255, 0),   (191, 255, 0),   (255, 255, 0),
    (0, 255, 63),  (63, 255, 63),  (127, 255, 63),  (191, 255, 63),  (255, 255, 63),
    (0, 255, 127), (63, 255, 127), (127, 255, 127), (191, 255, 127), (255, 255, 127),
    (0, 255, 191), (63, 255, 191), (127, 255, 191), (191, 255, 191), (255, 255, 191),
    (0, 255, 255), (63, 255, 255), (127, 255, 255), (191, 255, 255), (255, 255, 255)
]

def draw_fps(img, fps):
    """
    description: Draw fps on output image.
    param:
        img:    a opencv image object
        fps:    fps value for overall process
    return:
        no return
    """
    tl = (
        round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = [0, 0, 0]
    #c1, c2 = (0, 0), (101, 101)
    #cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    tf = max(tl - 1, 1)  # font thickness
    label = "fps: " + str(float("{:.2f}".format(fps)));
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c1 = (0, t_size[1] + 15)
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 15
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

    c1 = (0, t_size[1] + 3)
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] + 3
    cv2.putText(
        img,
        label,
        (c1[0], c1[1]),
        0,
        tl / 3,
        [225, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    
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

def save_json(self):
    # Dump Json
    with open('./result.json', 'w') as json_file:
        json.dump(self.annots, json_file, indent=2)

class ThreadTRT(threading.Thread):
    def __init__(self, pre_proc, infer_proc, post_proc, input_img, batch_size, annots, save_name, enable_draw_box, enable_write_json):
        threading.Thread.__init__(self)
        self.pre_proc = pre_proc
        self.infer_proc = infer_proc
        self.post_proc = post_proc
        self.input_img = input_img
        self.batch_size = batch_size
        self.annots = annots
        self.save_name = save_name
        
        self.enable_draw_box = enable_draw_box
        self.enable_write_json = enable_write_json

    def run(self):
        self.pre_proc.preprocess_image(self.input_img, self.infer_proc.get_infer_ptr())
        infer_result = self.infer_proc.inference()
        result_boxes, result_scores, result_classid = self.post_proc.post_process(infer_result)
        time = self.post_proc.post_process.proc_time

        if self.enable_write_output == True:
            for i in range(len(result_boxes)):
                box = result_boxes[i]
                plot_one_box(
                    box,
                    self.input_img,
                    label="{}:{:.2f}".format(
                        categories[int(result_classid[i])], result_scores[i]
                    ),
                )
            
            save_name = "output/0000_V0000_%03d.jpg" % index
            cv2.imwrite(save_name, self.input_img)
            self.add_img_annot(save_name, result_boxes, result_scores, time)

    def add_img_annot(self, file_name, boxes, scores, time):
        # Sum inference time
        self.annots['inference_time'] = self.annots['inference_time'] + time

        # Store annotation per image
        annotation = {'file_name': file_name, 'objects': []}
        for b, p in zip(boxes.tolist(), scores.tolist()):
            tmp = {'position': b, 'confidence_score': float(p)}
        annotation['objects'].append(tmp)
        self.annots['annotations'].append(annotation)
    
    def join(self):
        threading.Thread.join(self)
        return self.input_img, self.annots

def main():
    if ENABLE_CAMERA_LIVE is True:
        if len(sys.argv) == 1 :
            serial_number = 22246603
        else:
            serial_number = int(sys.argv[1])
        camera_wrapper = CameraZED(serial_number, INPUT_WIDTH, INPUT_HEIGHT, INPUT_FPS)
    
    pre_process_wrapper = PreProcessor((1080, 1920, 3), (608, 608, 3), ENABLE_TIME_PROFILE)
    inference_trt_wrapper = InferenceTRT("yolov5s.engine", ENABLE_TIME_PROFILE)
    post_process_wrapper = PostProcessor((1080, 1920, 3), (608, 608, 3), CONF_THRESH, IOU_THRESHOLD, ENABLE_TIME_PROFILE)
    
    if ENABLE_DUMMY_INPUT == True:
        for index in range(0, 10):
            if ENABLE_CAMERA_LIVE is True:
                h_img = camera_wrapper.capture_left()
            else:
                image_path = "image/0000_V0000_000.jpg"
                h_img = cv2.imread(image_path)
            
            thread1 = ThreadTRT(pre_process_wrapper, inference_trt_wrapper, post_process_wrapper, h_img, False)
            thread1.start()
            thread1.join()
        
        pre_process_wrapper.proc_time = 0
        inference_trt_wrapper.proc_time = 0
        post_process_wrapper.proc_time = 0
    
    start_frame = 0
    end_frame = 1000
    total_frame = 0
    
    for index in range(start_frame, end_frame):
        if ENABLE_CAMERA_LIVE is True:
            h_img = camera_wrapper.capture_left()
            total_frame += 1
        else:
            image_path = "image/0000_V0000_%03d.jpg" % index
            print(image_path)
            
            h_img = cv2.imread(image_path)
            total_frame += 1
        
        thread1 = ThreadTRT(pre_process_wrapper, inference_trt_wrapper, post_process_wrapper, h_img, ENABLE_WRITE_OUTPUT)
        thread1.start()
        out_img, annots = thread1.join()
        
        if ENABLE_DRAW_FPS is True:
            fps = 1000 / (pre_process_wrapper.proc_time + inference_trt_wrapper.proc_time + post_process_wrapper.proc_time)
            draw_fps(out_img, fps)
        
        if ENABLE_SHOW_OUTPUT is True:
            cv2.imshow("result", out_img)
            cv2.waitKey(1)
        
        if ENABLE_WRITE_OUTPUT is True:
            save_name = "output/%04d.jpg" % index
            cv2.imwrite(save_name, out_img)
        
        pre_process_total_time += pre_process_wrapper.proc_time
        inference_total_time += inference_trt_wrapper.proc_time
        post_process_total_time += post_process_wrapper.proc_time

        pre_process_wrapper.proc_time = 0
        inference_trt_wrapper.proc_time = 0
        post_process_wrapper.proc_time = 0

    if ENABLE_TIME_PROFILE is True:
        print("\n")
        print("Total frame            : ", total_frame)
        print("Avg. Pre-process time  : ", pre_process_total_time / total_frame, " msec")
        print("Avg. Inference time    : ", inference_total_time / total_frame, " msec")
        print("Avg. Post-process time : ", post_process_total_time / total_frame, " msec", "\n")
    
    if ENABLE_CAMERA_LIVE is True:
        camera_wrapper.destroy()
    
    pre_process_wrapper.destroy()
    inference_trt_wrapper.destroy()
    post_process_wrapper.destroy()

if __name__ == "__main__":
    main()
