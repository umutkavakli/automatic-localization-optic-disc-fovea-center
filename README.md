# Automatic Localization of Optic Disc and Focea Center Points

This project aims to detect the center of optic disc and fovea using only image processing techniques (no machine learning) in fundus images. Moreover, the segmentation of optic disc has been done for the purpose of the localization.

## 1) Dataset

Indian Diabetic Retinopathy Image Dataset (<b>[IDRID](https://idrid.grand-challenge.org/Home/)</b>) is used for this project.

<p align="center"><img src=".examples/image1.jpg" width="70%"></p>

The images of this dataset have a resolution of <b>4288×2848</b> pixels and are stored in JPG file format. Moreover, the dataset covers three distinct subsets of problems: Localization, Segmentation and Disease Grading. I only focused on localization and segmentation problems for this study. 

a) Ground Truth of Fovea Localization          |  b) Ground Truth of Optic Disc Localization
:-------------------------:|:-------------------------:
![](.examples/image2.jpg)  |  ![](.examples/image3.jpg)


## 2) Processing Steps
<b>Localization</b> subset contains <b>413</b> sample images for training and <b>103</b> sample images for testing by including the center location of both the optic disc and fovea in CSV file format. Since this study does not address model training, I only used the test subset. 

<b>Segmentation</b> subset of the dataset contains <b>54</b> training and <b>27</b> testing samples with corresponding segmentation masks. For this problem, <b>U-Net</b>, one of the well-known image segmentation algorithms in deep learning, was used to compare the performance of image processing methods. 

### 2.1) Optic Disc Localization
As an approach, I realized that the brightest parts of the images are mostly the areas where the optic disc is located. Therefore, I applied the histogram equalization algorithm to increase contrast and make the optic disc brighter after converting color space of the image from RGB to grayscale. 

Since the brightest parts are closer to the white value (255), I applied binary thresholding with a predefined value of 250. In this step, I iterate this process by lowering the threshold value until finding an object of sufficient area size. Sometimes small regions exceed the threshold and create noise in the segmentation, which I prevented by applying an erosion operation. This algorithm also uses a heap (priority queue) data structure to store the biggest objects because optic discs have a huge space in the images.


a) Input Image         |  b) Grayscale Image
:-------------------------:|:-------------------------:
![](.examples/image10.jpg)  |  ![](.examples/image11.jpg)

c) Histogram Eqaulized Image        |  d) Thresholded Image
:-------------------------:|:-------------------------:
![](.examples/image12.jpg)  |  ![](.examples/image13.jpg)


### 2.2) Fovea Localization

Both optic disc and fovea detection require similar processing steps to achieve meaningful results. Therefore, I applied most of the same methods with small adjustments. Because the fovea usually represents a black area in images, it is difficult to distinguish it from other parts of the image, even with the human eye. Thus, to convert dark pixels into light ones, I first convert the images into negative ones.

<br>

After analyzing images,  the fovea is always close to the center of the fundus. For this reason, I applied a circle masking operation to extract the middle part of the image as a region of interest with a predefined radius value. Since the image is in RGB color space, I transformed it into a grayscale and applied histogram equalization to sharpen the brightness of the fovea. The remaining manipulations are the same as the optic disc detection algorithm: thresholding, erosion, storing the largest contours in a heap and finding the center of the area.


a) Input Image         |  b) Negative Image         | c)  Grayscale Masked Image
:-------------------------:|:-------------------------:|:-------------------------:
![](.examples/image4.jpg)  |  ![](.examples/image5.jpg) | ![](.examples/image6.jpg) 

d) Histogram Eqaulized Image       |  e) Thresholded Image        | f) Erosion Applied Image
:-------------------------:|:-------------------------:|:-------------------------:
![](.examples/image7.jpg)  |  ![](.examples/image8.jpg) | ![](.examples/image9.jpg) 



## 3) Experiments and Results

### 3.1) Human Level Performance

<ul>
    <li> I observed that this region is sometimes very challenging to detect,
<b>even for humans</b>.</li>
    <li>Therefore, I created a simple application with a user interface for human
subjects to test their performance to set a threshold for our image
processing pipeline. </li>
</ul>

a) Easy Example          |  b) Hard Example
:-------------------------:|:-------------------------:
![](.examples/image14.jpg)  |  ![](.examples/image15.jpg)


### 3.2) Results of Fovea Localization in F1 Score

I have used F1 score for our metrics but this metric is designed for classification problem. Therefore, I made some adjustments to use it for my problems. I added a distance factor <b>d</b> to accept prediction true if the model find a coordinate which is equal or smaller than this factor. I have used this factor for both fovea and optic disc localization.
 
<div align="center">

| Model |d = 100 |d = 200 |d = 300 |d = 400 |d = 500 |
|-------|:------:|-------:|-------:|-------:|-------:|
| Human | 79.61% | 81.52% | 90.61% | 95.47% | 100%   |
| Ours  | 53.46% | 77.67% | 86.41% | 87.38% | 93.20% | 


</div>


### 3.3) Results of Optic Disc Localization in F1 Score

<div align="center">

| Model |d = 100 |d = 150 |d = 200 |d = 250 |d = 300 |
|-------|:------:|-------:|-------:|-------:|-------:|
| Ours  | 75.73% | 94.17% |	96.12% | 96.12% | 97.09% | 


</div>


### 3.4) Results of Optic Disc Segmentation in Dice Score

As I mentioned, I have used U-Net image segmentation algortithm to compare the performances of my model and a well-known deep learning model. I have used 54 training samples from dataset to train the U-Net model and 27 testing samples to measure the performance in dice score. You can see the hyperparameters I have used for U-Net algorithm: 

<div align="center">

| Hyperparameter | Value |
|-------|:------:|
| Input Size  |512x256 | 
| Learning Rate  | 10^-3 | 
| Loss Function  | Binary Cross-Entropy | 
| Optimizer  | Adam | 
| Batch Size  | 6 | 
| Epochs  | 150 | 


</div>

<br>
<br>

In the table below, I show overall performance for my image processing model and the U-Net algorithm. Since my model is not trained with any image, its preliminary results are promising.

<div align="center">

| Model | Dice Score |
|-------|:------:|
| U-Net  | 85.86%|
| Ours  | 72.55%|

</div>

<br>
<br>

 Ours      |  U-Net      | Ground Truth
:-------------------------:|:-------------------------:|:-------------------------:
![](.examples/image16.jpg)  |  ![](.examples/image17.jpg) | ![](.examples/image18.jpg) 

Ours    | U-Net       | Ground Truth
:-------------------------:|:-------------------------:|:-------------------------:
![](.examples/image19.jpg)  |  ![](.examples/image20.jpg) | ![](.examples/image21.jpg) 

