# Deep Learning Libraries based Rest API

### Deep Learning Libraries:
- MTCNN face detection in TensorFlow
- CTPN text detection in TensorFlow, Caffe
- CRNN text recognition in PyTorch
- Faster RCNN object detection in TensorFlow
- YOLO in TensorFlow (DarkFlow)
- DeepSort multiple person tracking in TensorFlow
- Softmax and Triplet person re-identification in TensorFlow

### Rest API
- Person detection and Bibnumber detection in Marathon

### Run Demo:

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

