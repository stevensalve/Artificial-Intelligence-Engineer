# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project - Credit Card Fraud Prediction

**Note (Written by Udacity)**

The Capstone is a two-staged project. The first is the proposal component, where you can receive valuable feedback about your project idea, design, and proposed solution. This must be completed prior to your implementation and submitting for the capstone project. 

You can find the [capstone proposal rubric here](https://review.udacity.com/#!/rubrics/410/view), and the [capstone project rubric here](https://review.udacity.com/#!/rubrics/108/view). Please ensure that you are following directions correctly before submitting these two stages which encapsulate your capstone.

Please email [machine-support@udacity.com](mailto:machine-support@udacity.com) if you have any questions.


## Summary

The problem chosen for this project is to predict fraudulent credit card transactions by using machine learning models. The models are going to be trained using supervised learning. A dataset containing thousands of individual transactions and their respective labels was obtained from [Kaggle website](https://www.kaggle.com/mlg-ulb/creditcardfraud).

The dataset contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA
transformation. Unfortunately, due to confidentiality issues, we cannot provide
the original features and more background information about the data. Features
V1, V2, ... V28 are the principal components obtained with PCA, the only features
which have not been transformed with PCA are 'Time' and 'Amount'. Feature
'Time' contains the seconds elapsed between each transaction and the first
transaction in the dataset. The feature 'Amount' is the transaction Amount, this
feature can be used for example-dependant cost-senstive learning. Feature 'Class'
represents the class labelling, it takes value 1 in case of fraud and 0 otherwise.

The objective was to create simple and commonly used machine learning models like logistic regression, KNN, random
forest and maybe others to compare how they perform regarding the metric
chosen (AUC) for the task of predicting fraudulent credit card transactions. After
that, I created an ensemble model that combines the predictions provided by the simple models as a way of further enhancing the performance.

## Main files

- Capstone Proposal - Alexandre Daltro.pdf: Project proposal before implementing it
- Udacity Review - Capstone Proposal.pdf: Proposal review by Udacity
- Capstone Project - Credit Card Fraud Prediction - Alexandre Daltro.pdf: Project report
- Capstone Project.ipynb: Complete project developed in Jupyter Notebook
- Udacity Review - Capstone Project.pdf: Project review by Udacity
- creditcard.rar: Dataset used in this project

## Installing Python packages

Python is available for all three major operating systems — Microsoft Windows, macOS, and Linux — and the installer, as well as the documentation, can be downloaded from the official Python website: https://www.python.org.

This book is written for Python version `>= 3.7.0`, and it is recommended
you use the most recent version of Python 3 that is currently available,
although most of the code examples may also be compatible with older versions of Python 3 and Python `>= 2.7.10`. If you decide to use Python 2.7 to execute the code examples, please make sure that you know about the major differences between the two Python versions. A good summary about the differences between Python 3 and 2.7 can be found at https://wiki.python.org/moin/Python2orPython3.

**Note**

You can check your current default version of Python by executing

    $ python -V


#### Pip

The additional packages that we will be using throughout this book can be installed via the `pip` installer program, which has been part of the Python standard library since Python 3.3. More information about pip can be found at https://docs.python.org/3/installing/index.html.

After we have successfully installed Python, we can execute pip from the command line terminal to install additional Python packages:

    pip install SomePackage


(where `SomePackage` is a placeholder for numpy, pandas, matplotlib, scikit-learn, and so forth).

Already installed packages can be updated via the `--upgrade` flag:

    pip install SomePackage --upgrade


#### Anaconda

A highly recommended alternative Python distribution for scientific computing
is Anaconda by Continuum Analytics. Anaconda is a free—including commercial use—enterprise-ready Python distribution that bundles all the essential Python packages for data science, math, and engineering in one user-friendly cross-platform distribution. The Anaconda installer can be downloaded at http://continuum.io/downloads#py34, and an Anaconda quick start-guide is available at https://store.continuum.io/static/img/Anaconda-Quickstart.pdf.

After successfully installing Anaconda, we can install new Python packages using the following command:

    conda install SomePackage

Existing packages can be updated using the following command:

    conda update SomePackage

Throughout this book, we will mainly use NumPy's multi-dimensional arrays to store and manipulate data. Occasionally, we will make use of pandas, which is a library built on top of NumPy that provides additional higher level data manipulation tools that make working with tabular data even more convenient. To augment our learning experience and visualize quantitative data, which is often extremely useful to intuitively make sense of it, we will use the very customizable matplotlib library.

#### Core packages

The version numbers of the major Python packages that were used for writing this book are listed below. Please make sure that the version numbers of your installed packages are equal to, or greater than, those version numbers to ensure the code examples run correctly:

- [Python](https://www.python.org/) >= 3.7.0
- [NumPy](http://www.numpy.org) >= 1.15.4
- [pandas](http://pandas.pydata.org) >= 0.24.1
- [SciPy](http://www.scipy.org) >= 1.2.0
- [scikit-learn](http://scikit-learn.org/stable/) >= 0.20.2
- [matplotlib](http://matplotlib.org) >= 3.0.2
- [seaborn](https://seaborn.pydata.org/) >= 0.9.0
- [imbalanced-learn](https://pypi.org/project/imbalanced-learn/) >= 0.4.3

## Python/Jupyter Notebook

Some readers might wonder about the `.ipynb` of the code files -- these files are IPython notebooks. I chose IPython notebooks over plain Python `.py` scripts, because I think that they are just great for data analysis projects! IPython notebooks allow us to have everything in one place: Our code, the results from executing the code, plots of our data, and documentation that supports the handy Markdown and powerful LaTeX syntax!

**Side Note:**  "IPython Notebook" recently became the "[Jupyter Notebook](<http://jupyter.org>)"; Jupyter is an umbrella project that aims to support other languages in addition to Python including Julia, R, and many more. Don't worry, though, for a Python user, there's only a difference in terminology (we say "Jupyter Notebook" now instead of "IPython Notebook").

The Jupyter notebook can be installed as usually via pip.

    $ pip install jupyter notebook

Alternatively, we can use the Conda installer if we have Anaconda or Miniconda installed:

    $ conda install jupyter notebook