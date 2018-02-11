# Deep Learning Libraries based Rest API

### Deep Learning Libraries (DLL):
- Modified SqueezeDet Multi-Task Learning (MTL) for Face detection, Landmark, Age, and Pose in TensorFlow
- MTCNN face detection in TensorFlow
- CTPN text detection in TensorFlow, Caffe
- CRNN text recognition in PyTorch
- Faster RCNN object detection in TensorFlow
- YOLO object detection in TensorFlow (DarkFlow)
- DeepSort multiple person tracking in TensorFlow
- Softmax and Triplet person re-identification in TensorFlow

### Example application using DLL:
- Tracking by Re-Identification
- Face detection in RTSP stream

### Rest API:
- Person detection and Bibnumber detection in Marathon

### Run Demo:
Examples using DLL: Please see the examples folder

Rest API:

```sh
DLL-RAPI$ sudo systemctl start apache2
```

```sh
DLL-RAPI$ redis-server
```
 
```sh
DLL-RAPI$ python model_RAPI.py
```

```sh
DLL-RAPI$ curl -X POST -F "image=@samples/000001.png" -F "rfid=@samples/rfid.csv" 'http://localhost/predict_bibnumber'
```

