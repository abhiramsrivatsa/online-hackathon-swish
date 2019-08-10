"""
1) Download and install Python 3 from official Python Language website
  https://python.org
2) Install the following dependencies via pip:
i. Tensorflow
  pip install tensorflow
ii. Numpy
  pip install numpy
iii. SciPy
  pip install scipy
iv. OpenCV
  pip install opencv-python
v. Pillow
  pip install pillow
vi. Matplotlib
  pip install matplotlib
vii. H5py
  pip install h5py
viii. Keras
  pip install keras
ix. ImageAI
  pip3 install imageai --upgrade

Download the RetinaNet model file that will be used for object detection via this
https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5

Copy the RetinaNet model file and the image you want to detect to the folder that contains the python file.
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
