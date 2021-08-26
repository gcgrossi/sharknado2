
<img src="assets/sharknado2_Cover.png">

# sharknado2 🦈
## Shar Detection with Python, OpenCV and Keras

[![](https://img.shields.io/badge/Python-B5a300?style=for-the-badge&logo=python)]()
[![](https://img.shields.io/badge/opencv-110354?style=for-the-badge&logo=opencv)]()
[![](https://img.shields.io/badge/keras-C10316?style=for-the-badge&logo=keras)]()

### Introduction

The first episode of sharknado was a classification model that was able to distinguish different shark species with an acceptable precision. The state of the art model was a finetuned VGG16 Convolutional network, trained on roughly 1000 images of different shark species with an out-ofthe-sample accuracy of 92%. 

At this point, a natural continuation of this serie seem to be to extend the model to a shark detection. While a shark classification can only tell if an image contains a sharks and what kind of shark is (so basically classify the topic of the image), a shark detector can also locate the shark in the image and predict the coordinates of the rectangle surrounding the detected object (the bounding box).
