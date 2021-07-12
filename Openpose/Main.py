"""
File: Main.py

Authors: Jinwoo Jeong <jw.jeong@keti.re.kr>
         Sungjei Kim <sungjei.kim@keti.re.kr>
         Seungho Lee <seunghl@keti.re.kr>

The property of program is under Korea Electronics Technology Institute.
For more information, contact us at <jw.jeong@keti.re.kr>.
"""

import sys
import cv2
import json

import trt_pose.coco
import trt_pose.models

from CameraZED import CameraZED

ENABLE_DRAW_FPS = True

INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080
INPUT_FPS = 30

INFER_WIDTH = 224
INFER_HEIGHT = 224

TOTAL_FRAME = 1000

ORIG_MODEL_PATH = "resnet18_baseline_att_224x224_A_epoch_249.pth"
TRT_MODEL_PATH = "resnet18_baseline_att_224x224_A_epoch_249_trt.pth"

def draw_fps(img, fps):
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

def preprocess(image):

def main():
    if len(sys.argv) == 1 :
        serial_number = 22246603
    else:
        serial_number = int(sys.argv[1])
    camera_wrapper = CameraZED(serial_number, INPUT_WIDTH, INPUT_HEIGHT, INPUT_FPS)
    
    with open ('human_pose.json', 'r') as f:
        human_pose = json.load(f)
    topology = trt_pose.coco.coco_category_to_topology(human_pose)
    
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    
    if os.path.exists(TRT_MODEL_PATH):
        OPTIMIZED_MODEL = TRT_MODEL_PATH
        model_trt = torch2trt.TRTModule()
        model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
    else:
        model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
        MODEL_WEIGHTS = ORIG_MODEL_PATH
        model.load_state_dict(torch.load(MODEL_WEIGHTS))
        
        data = torch.zeros((1, 3, INFER_HEIGHT, INFER_WIDTH)).cuda()
        model_trt = torch2trt.torch2trt(model, [data], fp16_mode = True, max_workspace_size=1<<25)
        
        OPTIMIZED_MODEL = TRT_MODEL_PATH
        torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
    
    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    
    for index in range(0, TOTAL_FRAME):
        capture_img = camera_wrapper.capture_left()
        
        cv2.imshow("result", capture_img)
        if cv2.waitKey(1) >= 0:
            break

if __name__ == "__main__":
    main()