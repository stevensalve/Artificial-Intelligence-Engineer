<h1>Artificial Intelligence Engineer Assignments </h1>
<h3>Deep Learning with Keras and Tensorflow</h3> <br>
<b>Assignments - Lending Club Loan Data Analysis</b><br>
<hr>
<div _ngcontent-ndh-c6="" class="tab-content ng-star-inserted" id="project-tab-content">
<div _ngcontent-ndh-c6="" class="project-info scrolly" >
<h3>Project 2  : Lending Club Loan Data Analysis</h3>
<div _ngcontent-ndh-c6="" class="ng-star-inserted"><div _ngcontent-ndh-c6="" class="project-information">
<div _ngcontent-ndh-c6="" class="project-description sl-ck-editor"><p _ngcontent-ndh-c6="">DESCRIPTION</p><div _ngcontent-ndh-c6=""><p>Create a model that predicts whether or not a loan will be default using the historical data.</p>

<p><strong>Problem Statement:</strong>&nbsp;&nbsp;</p>

<p>For companies like Lending Club correctly predicting whether or not a loan will be a default is very important. In this project, using the historical data from 2007 to 2015, you have to build a deep learning model to predict the chance of default for future loans. As you will see later this dataset is highly imbalanced and includes a lot of features that make this problem more challenging.</p>

<p><strong>Domain:</strong> Finance</p>

<p>Analysis to be done: Perform data preprocessing and build a deep learning prediction model.&nbsp;</p>

<p><strong>Content:&nbsp;</strong></p>

<p>Dataset columns and definition:</p>

<p>&nbsp;</p>

<ul>
	<li>
	<p><strong>credit.policy:</strong> 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.</p>
	</li>
	<li>
	<p><strong>purpose:</strong> The purpose of the loan (takes values "credit_card", "debt_consolidation", "educational", "major_purchase", "small_business", and "all_other").</p>
	</li>
	<li>
	<p><strong>int.rate:</strong> The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.</p>
	</li>
	<li>
	<p><strong>installment:</strong> The monthly installments owed by the borrower if the loan is funded.</p>
	</li>
	<li>
	<p><strong>log.annual.inc:</strong> The natural log of the self-reported annual income of the borrower.</p>
	</li>
	<li>
	<p><strong>dti:</strong> The debt-to-income ratio of the borrower (amount of debt divided by annual income).</p>
	</li>
	<li>
	<p><strong>fico:</strong> The FICO credit score of the borrower.</p>
	</li>
	<li>
	<p><strong>days.with.cr.line:</strong> The number of days the borrower has had a credit line.</p>
	</li>
	<li>
	<p><strong>revol.bal:</strong> The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).</p>
	</li>
	<li>
	<p><strong>revol.util: </strong>The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).</p>
	</li>
	<li>
	<p><strong>inq.last.6mths:</strong> The borrower's number of inquiries by creditors in the last 6 months.</p>
	</li>
	<li>
	<p><strong>delinq.2yrs:</strong> The number of times the borrower had been 30+ days past due on a payment in the past 2 years.</p>
	</li>
	<li>
	<p><strong>pub.rec: </strong>The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).</p>
	</li>
</ul>

<p>&nbsp;</p>

<p><strong>Steps to perform:</strong></p>

<p>Perform exploratory data analysis and feature engineering and then apply feature engineering. Follow up with a deep learning model to predict whether or not the loan will be default using the historical data.</p>

<p><strong>Tasks:</strong></p>

<p>1. &nbsp; &nbsp; Feature Transformation</p>

<ul>
	<li>
	<p>Transform categorical values into numerical values (discrete)</p>
	</li>
</ul>

<p>2. &nbsp; &nbsp; Exploratory data analysis of different factors of the dataset.</p>

<p>3. &nbsp; &nbsp; Additional Feature Engineering</p>

<ul>
	<li>
	<p>You will check the correlation between features and will drop those features which have a strong correlation</p>
	</li>
	<li>
	<p>This will help reduce the number of features and will leave you with the most relevant features</p>
	</li>
</ul>
<p>4. &nbsp; &nbsp; Modeling</p>
<ul>
	<li>
	<p>After applying EDA and feature engineering, you are now ready to build the predictive models</p>
	</li>
	<li>
	<p>In this part, you will create a deep learning model using Keras with Tensorflow backend</p>
	</li>
</ul>
</div></div>

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
https://www.kaggle.com/janiobachmann/lending-club-risk-analysis-and-metrics/notebook<br>

https://www.kaggle.com/wendykan/lending-club-loan-data <br>

http://rstudio-pubs-static.s3.amazonaws.com/290261_676d9bb194ae4c9882f599e7c0a808f2.html<br>

https://github.com/jalexander03/100119-Lending-Club-Loan-Data<br>

https://github.com/akshayr89/Lending-Club---Exploratory-Data-Analysis<br>

Differentt type of dataset: https://www.kaggle.com/wendykan/lending-club-loan-data<br>

https://github.com/harishpuvvada/LoanDefault-Prediction<br>

</p>