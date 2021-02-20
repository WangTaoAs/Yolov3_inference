# Yolov3_inference
***
## 1.Prepare the Pretrained weights
Downloading the Pre-trained Weights
Download the weights file into your detector directory ```./weights```. Grab the weights file from <a href="https://pjreddie.com/media/files/yolov3.weights">Here</a>. Or if you're on linux:
  ``` wget https://pjreddie.com/media/files/yolov3.weights```

## 2.Inference Parase
```--images``` :Image / Directory containing images to perform detection upon  
```--det``` :Image / Directory to store detection to  
```--bs```:batch size  
```--cfg```:Config file  
```--weights```:weights  file  

## 3.Run the code  

 ```python detector.py --det <Directory to store> --images <Directory containing images or just image> --bs <batch_size> --weights <weights_file>```
