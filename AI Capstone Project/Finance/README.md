<h1>Artificial Intelligence Engineer Assignments </h1>
<h3>AI Capstone Project</h3>
<b>Assignments : Finance</b><br>

<div _ngcontent-ppr-c44="" class="ng-star-inserted"><div _ngcontent-ppr-c44="" class="project-information"><div _ngcontent-ppr-c44="" class="project-description sl-ck-editor">
<div _ngcontent-ppr-c44="">

<p><strong>Problem Statement</strong></p>

<ul>
	<li>Finance Industry is the biggest consumer of Data Scientists. It faces constant attack by fraudsters, who try to trick the system. Correctly identifying fraudulent transactions is often compared with finding needle in a haystack because of the low event rate.&nbsp;</li>
	<li>It is important that credit card companies are able to recognize fraudulent credit card transactions so that the customers are not charged for items that they did not purchase.</li>
	<li>You are required to try various techniques such as supervised models with oversampling, unsupervised anomaly detection, and heuristics to get good accuracy at fraud detection.</li>
</ul>

<p><strong>Dataset Snapshot</strong></p>

<p>The datasets contain transactions made by credit cards in September 2013 by European cardholders. This dataset represents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.</p>

<p>&nbsp;</p>

<p><img alt="" height="80" src="https://cfs22.simplicdn.net/paperclip/project/images/1566546571_cap 2.png" width="1262"></p>

<p>&nbsp;</p>

<p>It contains only numerical input variables which are the result of a PCA transformation.&nbsp;<br>
Features V1, V2, ... V28 are the principal components obtained with PCA.&nbsp;<br>
The only features which have not been transformed with PCA are 'Time' and 'Amount'</p>

<p>&nbsp;</p>

<p><strong>Project Task: Week 1</strong></p>

<p><strong>Exploratory Data Analysis (EDA):</strong></p>

<p>1. &nbsp; &nbsp;Perform an EDA on the Dataset.<br>
&nbsp; &nbsp; &nbsp; &nbsp;a)&nbsp;&nbsp; &nbsp;Check all the latent features and parameters with their mean and standard deviation. Value are close to 0 centered (mean)<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; with unit standard deviation<br>
&nbsp; &nbsp; &nbsp; &nbsp;b)&nbsp;&nbsp; &nbsp;Find if there is any connection between Time, Amount, and the transaction being fraudulent.<br>
2. &nbsp; &nbsp;Check the class count for each class. It’s a class Imbalance problem.<br>
3. &nbsp; &nbsp;Use techniques like undersampling or oversampling before running Naïve Bayes, Logistic Regression or SVM.<br>
&nbsp; &nbsp; &nbsp; &nbsp;a.&nbsp;&nbsp; &nbsp;Oversampling or undersampling can be used to tackle the class imbalance problem<br>
&nbsp; &nbsp; &nbsp; &nbsp;b.&nbsp;&nbsp; &nbsp;Oversampling increases the prior probability of imbalanced class and in case of other classifiers, error gets multiplied as the&nbsp;<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; low-proportionate class is mimicked multiple times.<br>
4 &nbsp; &nbsp; Following are the matrices for evaluating the model performance: Precision, Recall, F1-Score, AUC-ROC curve. Use F1-Score as<br>
&nbsp; &nbsp; &nbsp; &nbsp;the evaluation criteria for this project.</p>

<p><strong>Project Task: Week 2</strong></p>

<p><strong>Modeling Techniques:</strong></p>

<p>Try out models like Naive Bayes, Logistic Regression or SVM. Find out which one performs the best<br>
Use different Tree-based classifiers like Random Forest and XGBoost.&nbsp;<br>
&nbsp; &nbsp; &nbsp; &nbsp;a.&nbsp;&nbsp; &nbsp;Remember Tree-based classifiers work on two ideologies: Bagging or Boosting<br>
&nbsp; &nbsp; &nbsp; &nbsp;b.&nbsp;&nbsp; &nbsp;Tree-based classifiers have fine-tuning parameters which takes care of the imbalanced class. Random-Forest and XGBboost.<br>
Compare the results of 1 with 2 and check if there is any incremental gain.</p>

<p><strong>Project Task: Week 3</strong></p>

<p><strong>Applying ANN:</strong></p>

<p>Use ANN (Artificial Neural Network) to identify fradulent&nbsp;and non-fradulent.<br>
&nbsp; &nbsp; &nbsp; &nbsp;a)&nbsp;&nbsp; &nbsp;Fine-tune number of layers<br>
&nbsp; &nbsp; &nbsp; &nbsp;b)&nbsp;&nbsp; &nbsp;Number of Neurons in each layers<br>
&nbsp; &nbsp; &nbsp; &nbsp;c)&nbsp;&nbsp; &nbsp;Experiment in batch-size<br>
&nbsp; &nbsp; &nbsp; &nbsp;d)&nbsp;&nbsp; &nbsp;Experiment with number of epochs. Check the observations in loss and accuracy<br>
&nbsp; &nbsp; &nbsp; &nbsp;e)&nbsp;&nbsp; &nbsp;Play with different Learning Rate variants of Gradient Descent like Adam, SGD, RMS-prop<br>
&nbsp; &nbsp; &nbsp; &nbsp;f) &nbsp; &nbsp;Find out which activation performs best for this use case and why?<br>
&nbsp; &nbsp; &nbsp; &nbsp;g)&nbsp;&nbsp; &nbsp;Check Confusion Matrix, Precision, Recall and F1-Score<br>
2. &nbsp; &nbsp;Try out Dropout for ANN. How is it performed? Compare model performance with the traditional ML based prediction models from<br>
&nbsp; &nbsp; &nbsp; &nbsp;above.&nbsp;<br>
3. &nbsp; &nbsp;Find the best setting of neural net that can be best classified as fraudulent and non-fraudulent transactions. Use<br>
&nbsp; &nbsp; &nbsp; &nbsp;techniques like Grid Search, Cross-Validation and Random search.</p>

<p><strong>Anomaly Detection:</strong></p>

<p>4. &nbsp; &nbsp; Implement anomaly detection algorithms.<br>
&nbsp; &nbsp; &nbsp; &nbsp; a)&nbsp;&nbsp; &nbsp;Assume that the data is coming from a single or a combination of multivariate Gaussian<br>
&nbsp; &nbsp; &nbsp; &nbsp; b)&nbsp;&nbsp; &nbsp;Formalize a scoring criterion, which gives a scoring probability for the given data point whether it belongs to the<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; multivariate Gaussian or Normal Distribution fitted in a)<br>
<strong>Project Task: Week 4</strong></p>

<p><strong>Inference and Observations:</strong></p>

<ol>
	<li>Visualize the scores for Fraudulent and Non-Fraudulent transactions.</li>
	<li>Find out the threshold value for marking or reporting a transaction as fraudulent in your anomaly detection system.</li>
	<li>Can this score be used as an engineered feature in the models developed previously? Are there any incremental gains in F1-Score? Why or Why not?</li>
	<li>Be as creative as possible in finding other interesting insights.</li>
</ol>

<p>Download the Data Sets from <a href="https://www.dropbox.com/s/6z5jxcqaqipxiun/Project%202-Finance-Datasets.zip?dl=0" target="_blank">here</a> .<br>
&nbsp;</p>
</div></div></div>
</div></app-project-footer></div>