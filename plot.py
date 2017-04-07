import matplotlib.pyplot as plt
import numpy as np 
import math
import itertools

def plot_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
   
    plt.xticks(tick_marks, classes, rotation=45)
    if title == 'Confusion matrix':
        plt.yticks(tick_marks, classes)
    elif title == 'Performance matrix':
        plt.yticks(np.arange(4), ['Precision', 'Recall', 'F1 score', 'g-measure'])

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if title == 'Confusion matrix':
            print("Normalized confusion matrix")
        elif title == 'Performance matrix':
            print("Normalized performance matrix")
    else:
        if title == 'Confusion matrix':
            print('Confusion matrix, without normalization')
        elif title == 'Performance matrix':
            print('Performance matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if title == 'Confusion matrix':
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    elif title == 'Performance matrix':
        plt.ylabel('Measurements')
        plt.xlabel('Classes')