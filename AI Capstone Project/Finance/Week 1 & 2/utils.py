import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LABELS = ["Normal", "Fraud"]


def plotConfusionMatrix(conf):
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, cmap="Greens", fmt='g', cbar_kws={'label': 'Number of Transactions'})
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def createBatches(dataToSplit, constData, batch):
    batchSize = int(dataToSplit.shape[0] / batch)
    dataToSplit = dataToSplit.ix[random.sample(list(dataToSplit.index), batch * batchSize)]
    batch_no_array = np.array([])

    for itr in range(1, batch + 1):
        batch_no_array = np.append(batch_no_array, [itr] * batchSize)

    np.random.shuffle(batch_no_array)
    dataToSplit.loc[:, 'batch'] = batch_no_array

    batches = []
    for itr in range(1, batch + 1):
        newBatch = pd.concat([dataToSplit[dataToSplit['batch'] == itr], constData], sort=False)
        newBatch.drop(['batch'], axis=1, inplace=True)
        batches.append(newBatch)

    return batches


def splitFraudNonFraud(data):
    nonFraud = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]

    return nonFraud, fraud