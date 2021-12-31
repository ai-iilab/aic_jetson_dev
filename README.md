# aic_jetson_dev

### Common
1. Set the profile.

       $ vim ~/.profile

   Write the following context and save file.

       export OPENBLAS_CORETYPE=ARMv8

   Apply the profile.
   
       $ source ~/.profile

2. Install PyTorch for NVIDIA Jetson. (https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl)

3. Install torchvision. (https://github.com/pytorch/vision/archive/refs/tags/v0.9.0.zip)

       $ unzip vision-0.9.0.zip
       $ cd vision-0.9.0
       $ export BUILD_VERSION=0.9.0
       $ python3 setup.py install --user
---
### Yolov5s
1. Install python packages.

       $ pip install -r requirements.txt

2. Run shell script.

       $ ./yolov5s_w0.04_d0.33_time.sh

3. Results - Inference speed (1 frame/ms). 
               
       |Batch|Python|TensorRT|
       |-----|------|--------|
       |1|26.69|15.96|
       |2|27.05|10.75|
       |4|17.41|6.95|
       |8|10.53|5.75|
       |16|8.13|5.34|
       
---
### Openpose
1. Install ZED SDK. (https://download.stereolabs.com/zedsdk/3.5/jp45/jetsons)
2. Install torch2trt. (https://github.com/NVIDIA-AI-IOT/torch2trt)
3. Install python packages.

       $ pip install -r requirements.txt

4. Connect ZED Camera to NVIDIA Jetson.

5. Run shell script.

       $ ./openpose.sh