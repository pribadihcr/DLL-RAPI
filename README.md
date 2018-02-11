# Deep Learning Libraries based Rest API

Run Demo:

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

