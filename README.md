# FruitDet: Attentive Feature Aggregation for Real-Time Fruit Detection in Orchards

This is an implementation of the FruitDet. A jupyter notebook version and a python version is attached in the repo.

## Abstract

Computer vision is currently experiencing success in various domains due to the harnessing of deep learning strategies. In the case of precision agriculture, computer vision is being investigated for detecting fruits from orchards. However, such strategies limit too-high complexity computation that is impossible to embed in an automated device. Nevertheless, most investigation of fruit detection is limited to a single fruit, resulting in the necessity of a one-to-many object detection system. This paper introduces a generic detection mechanism named FruitDet, designed to be prominent for detecting fruits. The FruitDet architecture is designed on the YOLO pipeline and achieves better performance in detecting fruits than any other detection model. The backbone of the detection model is implemented using DenseNet architecture. Further, the FruitDet is packed with newer concepts: attentive pooling, bottleneck spatial pyramid pooling, and blackout mechanism. The detection mechanism is benchmarked using five datasets, which combines a total of eight different fruit classes. The FruitDet architecture acquires better performance than any other recognized detection methods in fruit detection.


### Model Architecture

![model](https://github.com/QuwsarOhi/fruitdet/blob/master/imgs/model.png)


### Keywords:
* Deep learning
* Object detection
* Agriculture
* Convolutional Neural Network


If you use this work, please cite this article: 
https://doi.org/10.3390/agronomy11122440