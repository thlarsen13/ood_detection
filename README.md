# Preserving Calibration of Neural Networks under Distribution Shift


__Motivation.__ DNN classifiers trained on a single dataset tend to perform poorly on OOD test data.
Worse, these classifiers may report very high confidence for incorrect predictions. 
This overconfidence raises serious concerns of the application of ML to real world systems. 
The goal of this project is to develop algorithms to maintain the calibration of ML models after distribution shift. 

## Method 

Informally, we say that a model is calibrated if the predicted probabilities match with the observed chance that something is true. For example, if we have 100 samples for which our model predicted a .7 chance of being a certain class, the model is calibrated if close to 70 out of the 100. 

A natural way to measure calibration is Expected Calibration Error, which is the average difference between the true probability of being correct and the actual probability. To measure this, we need to split predictions with similar confidences into bins. The confidence of a bin is the prediction confidence of the items in the bin. The accuracy of a bin is:

<img width="249" alt="Screen Shot 2022-02-07 at 6 21 21 PM" src="https://user-images.githubusercontent.com/47545823/152888850-f7ace346-5281-4baa-bbff-b0432611eb3b.png">

The **Expected Calibration Error (ECE)** of a model is: 

<img width="240" alt="Screen Shot 2022-02-07 at 6 21 25 PM" src="https://user-images.githubusercontent.com/47545823/152888973-4786e7df-5345-461f-86d2-c325b8b672ed.png">

This suggests a natural objective function: 

<img width="240" alt="Screen Shot 2022-02-07 at 6 24 16 PM" src="https://user-images.githubusercontent.com/47545823/152889107-a3afd1e6-54d7-4fb4-aa56-41d9ba2fbadc.png">

which is normal cross entropy, with an additional term minimizing empirical deviation from perfect calibration. 

Models here were trained using Tensorflow 2.7.0, CUDA 11.0, and a GeForce RTX 2080 Ti GPU. 

## Preliminary Results 

We test this on MNIST using both a standard training using categorical cross entropy (CCE), and the proposed method (CCE + ECE). Here is an example of a shift that decreases the contrast 
of images from MNIST. 

![Example Shift](https://user-images.githubusercontent.com/47545823/152887788-f5cef621-799c-4df0-9532-c130230a5b63.png)


![Mnist Results](https://user-images.githubusercontent.com/47545823/152888270-e9cb59f7-a118-4b32-85da-813706e192b1.png)
Using the standard method, we see that as the magnitude of the shift, calibration decreases, as one would expect. However, with the proposed method, ECE is both significantly reduced, and the trend is less increasing. 
