<h1>Artificial Intelligence Engineer Assignments </h1>
<h3>Deep Learning with Keras and Tensorflow</h3> 
<b>Assignments - Pet Classification Model Using CNN.</b><br>
<br>
<hr>
<div _ngcontent-ndh-c6="" class="tab-content ng-star-inserted" id="project-tab-content">
<div _ngcontent-ndh-c6="" class="project-info scrolly" >
<h3>Project 3  : Pet Classification Model Using CNN.</h3>
<div _ngcontent-ndh-c6="" class="ng-star-inserted"><div _ngcontent-ndh-c6="" class="project-information">
<div _ngcontent-ndh-c6="" class="project-description sl-ck-editor"><p _ngcontent-ndh-c6="">DESCRIPTION</p><div _ngcontent-ndh-c6=""><p>Build a CNN model that classifies the given pet images correctly into dog and cat images.&nbsp;</p>

<p>The project scope document specifies the requirements for the project “Pet Classification Model Using CNN.” Apart from specifying the functional and nonfunctional requirements for the project, it also serves as an input for project scoping.&nbsp;</p>

<p><strong>Project Description and Scope&nbsp;</strong></p>

<p>You are provided with the following resources that can be used as inputs for your model:&nbsp;</p>

<p>1. A collection of images of pets, that is, cats and dogs. These images are of&nbsp;</p>

<p>different sizes with varied lighting conditions. 2. Code template containing the following code blocks:&nbsp;</p>

<p>a. Import modules (part 1) b. Set hyper parameters (part 2) c. Read image data set (part 3) d. Run TensorFlow model (part 4)&nbsp;</p>

<p>You are expected to write the code for CNN image classification model (between Parts 3 and 4) using TensorFlow that trains on the data and calculates the accuracy score on the test data.&nbsp;</p>

<p><strong>Project Guidelines</strong>&nbsp;</p>

<p>Begin by extracting ipynb file and the data in the same folder. The CNN model (cnn_model_fn) should have the following layers:&nbsp;</p>

<p>● Input layer&nbsp;</p>

<p>● Convolutional layer 1 with 32 filters of kernel size[5,5]&nbsp;</p>

<p>● Pooling layer 1 with pool size[2,2] and stride 2&nbsp;</p>

<p>● Convolutional layer 2 with 64 filters of kernel size[5,5]&nbsp;</p>

<p>● Pooling layer 2 with pool size[2,2] and stride 2&nbsp;</p>

<p>● Dense layer whose output size is fixed in the hyper parameter: fc_size=32&nbsp;</p>

<p>● Dropout layer with dropout probability 0.4&nbsp;</p>

<p>Predict the class by doing a softmax on the output of the dropout layers.&nbsp;</p>

<p>This should be followed by training and evaluation:&nbsp;</p>

<p>1 | Page ©Simplilearn. All rights reserved&nbsp;</p>
<p>● For the training step, define the loss function and minimize it&nbsp;</p>
<p>● For the evaluation step, calculate the accuracy&nbsp;</p>
<p>Run the program for 100, 200, and 300 iterations, respectively. Follow this by a report on the final accuracy and loss on the evaluation data.&nbsp;</p>
<p><strong>Prerequisites&nbsp;</strong></p>
<p>To execute this project, refer to the installation guide in the downloads section of LMS.&nbsp;</p>


<p>
Few Reference Links to Study before starting the project

https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

https://github.com/girishkuniyal/Cat-Dog-CNN-Classifier

https://www.kaggle.com/c/dogs-vs-cats/overview

https://github.com/coolioasjulio/Cat-Dog-CNN-Classifier

https://github.com/ardamavi/Dog-Cat-Classifier

https://www.kaggle.com/c/dogs-vs-cats/notebooks

https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial

https://www.kaggle.com/xiormeesh/cnn-cats-vs-dogs-classification

</p>