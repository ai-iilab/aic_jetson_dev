# aic_jetson_dev
---
### Yolov5s
1. Set the profile

       $ vim ~/.profile

   Write the following context and save file

       export OPENBLAS_CORETYPE=ARMv8

   $ source ~/.profile

2. Install PyTorch for NVIDIA Jetson (https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl)

3. Install torchvision (https://github.com/pytorch/vision/archive/refs/tags/v0.9.0.zip)

       $ unzip vision-0.9.0.zip
       $ cd vision-0.9.0
       $ export BUILD_VERSION=0.9.0
       $ python3 setup.py install --user

4. Install python packages

       $ pip install -r requirements.txt

5. Run shell script

       $ ./yolov5s_w0.04_d0.33_time.sh

---
### Openpose
1. Install ZED SDK (https://download.stereolabs.com/zedsdk/3.5/jp45/jetsons)
2. Install torch2trt (https://github.com/NVIDIA-AI-IOT/torch2trt)
3. Install python packages

       $ pip install -r requirements.txt

4. Connect ZED Camera to NVIDIA Jetson

5. Run shell script

       $ ./openpose.sh