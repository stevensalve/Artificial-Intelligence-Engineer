<h1>Artificial Intelligence Engineer Assignments </h1>
<h3>Module : AI Capstone Project</h3><br>
 
 <b>Assignments : Retail</b><br>

 <div _ngcontent-ppr-c44="" class="ng-star-inserted"><div _ngcontent-ppr-c44="" class="project-information"><div _ngcontent-ppr-c44="" class="project-description sl-ck-editor">
 
 <div _ngcontent-ppr-c44=""><p><strong>Problem Statement</strong></p>

<ul>
	<li>Demand Forecast is one of the key tasks in Supply Chain and Retail Domain in general. It is key in effective operation and optimization of retail supply chain. Effectively solving this problem requires knowledge about a wide range of tricks in Data Sciences and good understanding of ensemble techniques.&nbsp;</li>
	<li>You are required to predict sales for each Store-Day level for one month. All the features will be provided and actual sales that happened during that month will also be provided for model evaluation.&nbsp;</li>
</ul>

<p><strong>Dataset Snapshot</strong></p>

<p>Training Data Description: Historic sales at Store-Day level for about two years for a retail giant, for more than 1000 stores. Also, other sale influencers like, whether on a particular day the store was fully open or closed for renovation, holiday and special event details, are also provided.&nbsp;</p>

<p>&nbsp;</p>

<p><img alt="" height="175" src="1566547170_cap 3.png" width="662"></p>

<p><strong>Project Task: Week 1</strong></p>

<p><strong>Exploratory Data Analysis (EDA) and Linear Regression:</strong></p>

<p>1. &nbsp; &nbsp; &nbsp;Transform the variables by using data manipulation techniques like, One-Hot Encoding&nbsp;<br>
2. &nbsp; &nbsp; &nbsp;Perform an EDA (Exploratory Data Analysis) to see the impact of variables over Sales.<br>
3. &nbsp; &nbsp; &nbsp;Apply Linear Regression to predict the forecast and evaluate different accuracy metrices like RMSE (Root Mean Squared Error)<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;and MAE(Mean Absolute Error) and determine which metric makes more sense. Can there be a better accuracy metric?<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;a)&nbsp;&nbsp; &nbsp; &nbsp;Train a single model for all stores, using storeId as a feature.<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;b)&nbsp;&nbsp; &nbsp; &nbsp;Train separate model for each store.<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;c)&nbsp;&nbsp; &nbsp; &nbsp;Which performs better and Why? [In the first case, parameters are shared and not very free but not in second case]<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;d)&nbsp;&nbsp; &nbsp; &nbsp;Try Ensemble of b) and c). What are the findings?<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;e)&nbsp;&nbsp; &nbsp; &nbsp;Use Regularized Regression. It should perform better in an unseen test set. Any insights??<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;f)&nbsp;&nbsp; &nbsp; &nbsp;Open-ended modeling to get possible predictions.</p>

<p><strong>Project Task: Week 2</strong></p>

<p><strong>Other Regression Techniques:</strong></p>

<p>1. When store is closed, sales = 0. Can this insight be used for Data Cleaning? Perform this and retrain the model. Any benefits of this step?<br>
2. Use Non-Linear Regressors like Random Forest or other Tree-based Regressors.<br>
&nbsp; &nbsp; &nbsp; &nbsp;a)&nbsp;&nbsp; &nbsp;Train a single model for all stores, where storeId can be a feature.<br>
&nbsp; &nbsp; &nbsp; &nbsp;b)&nbsp;&nbsp; &nbsp;Train separate models for each store.<br>
&nbsp; &nbsp; &nbsp; &nbsp;Note: Dimensional Reduction techniques like, PCA and Tree’s Hyperparameter Tuning will be required. Cross-validate to find the<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; best parameters. Infer the performance of both the models.&nbsp;<br>
3 Compare the performance of Linear Model and Non-Linear Model from the previous observations. Which performs better and why?<br>
4. Train a Time-series model on the data taking time as the only feature. This will be a store-level training.<br>
&nbsp; &nbsp; &nbsp; &nbsp;a) &nbsp; &nbsp;Identify yearly trends and seasonal months<br>
&nbsp;</p>

<p><strong>Project Task: Week 3</strong></p>

<p><strong>Implementing Neural Networks:</strong></p>

<ol>
	<li>Train a LSTM on the same set of features and compare the result with traditional time-series model.</li>
	<li>Comment on the behavior of all the models you have built so far</li>
	<li>Cluster stores using sales and customer visits as features. Find out how many clusters or groups are possible. Also visualize the results.</li>
	<li>Is it possible to have separate prediction models for each cluster? Compare results with the previous models.</li>
</ol>

<p><strong>Project Task: Week 4</strong></p>

<p><strong>Applying ANN:</strong></p>

<p>1.&nbsp; &nbsp; &nbsp;Use ANN (Artificial Neural Network) to predict Store Sales.<br>
&nbsp; &nbsp; &nbsp; &nbsp;a)&nbsp;&nbsp; &nbsp;Fine-tune number of layers,<br>
&nbsp; &nbsp; &nbsp; &nbsp;b)&nbsp;&nbsp; &nbsp;Number of Neurons in each layers .<br>
&nbsp; &nbsp; &nbsp; &nbsp;c)&nbsp;&nbsp; &nbsp;Experiment in batch-size.<br>
&nbsp; &nbsp; &nbsp; &nbsp;d)&nbsp;&nbsp; &nbsp;Experiment with number of epochs. Carefully observe the loss and accuracy? What are the observations?<br>
&nbsp; &nbsp; &nbsp; &nbsp;e)&nbsp;&nbsp; &nbsp;Play with different &nbsp;Learning Rate &nbsp;variants of Gradient Descent like Adam, SGD, RMS-prop.<br>
&nbsp; &nbsp; &nbsp; &nbsp;f)&nbsp;&nbsp; &nbsp;Which activation performs best for this use case and why?<br>
&nbsp; &nbsp; &nbsp; &nbsp;g)&nbsp;&nbsp; &nbsp;Check how it performed in the dataset, calculate RMSE.<br>
2. &nbsp; &nbsp;Use Dropout for ANN and find the optimum number of clusters (clusters formed considering the features: sales and customer<br>
&nbsp; &nbsp; &nbsp; &nbsp;visits). Compare model performance with traditional ML based prediction models.&nbsp;<br>
3. &nbsp; &nbsp;Find the best setting of neural net that minimizes the loss and can predict the sales best. Use techniques like Grid<br>
&nbsp; &nbsp; &nbsp; &nbsp;search, cross-validation and Random search.</p>

<p>Downlod the data sets from <a href="https://github.com/Simplilearn-Edu/Artificial-Intelligence-Capstone-Project-Datasets" target="_blank">here</a> .</p>
</div></div></div>

</div>