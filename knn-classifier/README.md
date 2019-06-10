## 使用方法
```
usage: knn_classify.py [-h] [-fp IMAGENAME] [-vgg16 IMGBYVGG]
optional arguments:
  -h, --help       show this help message and exit
  -fp IMAGENAME    input filename and dealwith it plainly
  -vgg16 IMGBYVGG  input filename and extract features with vgg16
```

* -fp    [file plainly] just using hsvgrams to calculate distances
* -vgg16 [extract features] using keras.applications.vgg16 to extract features to calculate distances
*        [noparam] default using hsvgrams to predict testdataset and checkout accuracy 
