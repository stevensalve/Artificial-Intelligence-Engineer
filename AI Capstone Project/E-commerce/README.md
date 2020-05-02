<h1>Artificial Intelligence Engineer Assignments </h1>
<h3>AI Capstone Project</h3>
 <b>Assignments : E-commerce</b><br>
 <div _ngcontent-ppr-c44="" class="ng-star-inserted"><div _ngcontent-ppr-c44="" class="project-information"><div _ngcontent-ppr-c44="" class="project-description sl-ck-editor"><p _ngcontent-ppr-c44="">DESCRIPTION</p><div _ngcontent-ppr-c44=""><p><strong>Problem Statement</strong></p>
<ul>
	<li>Amazon is an online shopping website that now caters to millions of people everywhere. Over 34,000 consumer reviews for Amazon brand products like Kindle, Fire TV Stick and more are provided.&nbsp;</li>
	<li>The dataset has attributes like brand, categories, primary categories, reviews.title, reviews.text, and the sentiment. Sentiment is a categorical variable with three levels "Positive", "Negative“, and "Neutral". For a given unseen data, the sentiment needs to be predicted.</li>
	<li>You are required to predict Sentiment or Satisfaction of a purchase based on multiple features and review text.</li>
</ul>

<p><strong>Dataset Snapshot</strong></p>

<p><strong><img alt="" height="339" src="https://cfs22.simplicdn.net/paperclip/project/images/1566552102_Picture1.png" width="1179"></strong></p>

<p><strong>Project Task: Week 1</strong></p>

<p><strong>Class Imbalance Problem:</strong></p>

<p>1. Perform an EDA on the dataset.</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp;a) &nbsp;See what a positive, negative, and neutral review looks like</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp;b) &nbsp;Check the class count for each class. It’s a class imbalance problem.</p>

<p>2. Convert the reviews in Tf-Idf score.</p>

<p>3. Run multinomial Naive Bayes classifier. Everything will be classified as positive because of the class imbalance.</p>

<p><strong>Project Task: Week 2</strong></p>

<p><strong>Tackling Class Imbalance Problem:</strong></p>

<ol>
	<li>Oversampling or undersampling can be used to tackle the class imbalance problem.&nbsp;</li>
	<li>In case of class imbalance criteria, use the following metrices for evaluating model performance: precision, recall, F1-score, AUC-ROC curve. Use F1-Score as the evaluation criteria for this&nbsp; &nbsp; &nbsp; project.</li>
	<li>Use Tree-based classifiers like Random Forest and XGBoost.</li>
</ol>

<p>&nbsp; &nbsp; &nbsp; &nbsp;<strong>Note</strong>: Tree-based classifiers work on two ideologies namely, Bagging or Boosting and have fine-tuning parameter which takes care of the imbalanced class.</p>

<p><strong>Project Task: Week 3</strong></p>

<p><strong>Model Selection:</strong></p>

<ol>
	<li>Apply multi-class SVM’s and neural nets.</li>
	<li>Use possible ensemble techniques like: XGboost + oversampled_multinomial_NB.</li>
	<li>Assign a score to the sentence sentiment (engineer a feature called sentiment score). Use this engineered feature in the model and check for improvements. Draw insights on the same.</li>
</ol>

<p><strong>Project Task: Week 4</strong></p>

<p><strong>Applying LSTM:</strong></p>

<ol>
	<li>Use LSTM for the previous problem (use parameters of LSTM like top-word, embedding-length, Dropout, epochs, number of layers, etc.)</li>
</ol>

<p>&nbsp; &nbsp; &nbsp; &nbsp;<strong>Hint</strong>: Another variation of LSTM, GRU (Gated Recurrent Units) can be tried as well.</p>

<p>&nbsp; &nbsp; &nbsp; 2. Compare the accuracy of neural nets with traditional ML based algorithms.</p>

<p>&nbsp; &nbsp; &nbsp; 3. Find the best setting of LSTM (Neural Net) and GRU that can best classify the reviews as positive, negative, and neutral.&nbsp;</p>

<p>&nbsp; &nbsp; &nbsp; &nbsp;<strong>Hint</strong>: Use techniques like Grid Search, Cross-Validation and Random Search</p>

<p><strong>Optional Tasks: Week 4</strong></p>

<p><strong>Topic Modeling:</strong></p>

<p>&nbsp; &nbsp;1. Cluster similar reviews.<br>
&nbsp; &nbsp; &nbsp; &nbsp;<strong>Note</strong>: Some reviews may talk about the device as a gift-option. Other reviews may be about product looks and some may<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; highlight about its battery and performance. Try naming the clusters.<br>
&nbsp; &nbsp;2. Perform Topic Modeling<br>
&nbsp; &nbsp; &nbsp; &nbsp;<strong>Hint</strong>: Use scikit-learn provided Latent Dirchlette Allocation (LDA) and Non-Negative Matrix Factorization (NMF).</p>

<p>Download the Data sets from <a href="https://github.com/Simplilearn-Edu/Artificial-Intelligence-Capstone-Project-Datasets" target="_blank">here</a>&nbsp;.</p>
</div></div></div>
</div></app-project-footer></div>