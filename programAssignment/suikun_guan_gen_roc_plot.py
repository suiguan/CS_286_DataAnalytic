# This is for Programming Assignment 4 in CS 286
# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

#Imports
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc

#inputs:
#y_test: numpy array with ground truth
#y_score: numpy array with scores predicted from a classifier
#n_classes: total number of output classes in the dataset 
#specClass (optional): if want to plot specific class, then
#supply the class number here. Otherwise, it will plot for multiclass
#
#outputs: display the ROC plot
def displayROC(y_test, y_score, n_classes, specClass=None):
   # Compute ROC curve and ROC area for each class
   fpr = dict()
   tpr = dict()
   roc_auc = dict()
   for i in range(n_classes):
       fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
       roc_auc[i] = auc(fpr[i], tpr[i])

   # Compute micro-average ROC curve and ROC area
   fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
   roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

   #create plt figure
   plt.figure()
   lw = 2

   #either 1 or 2 below:

   #1. Plot of a ROC curve for a specific class
   if specClass != None and specClass < n_classes:
      plt.plot(fpr[2], tpr[2], color='darkorange',
               lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[specClass])
      plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver operating characteristic example for class %s' % specClass)
      plt.legend(loc=0) #pick best location
      plt.show()
      return

   #2. Plot ROC curves for the multiclass problem

   # Compute macro-average ROC curve and ROC area
   # First aggregate all false positive rates
   all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

   # Then interpolate all ROC curves at this points
   mean_tpr = np.zeros_like(all_fpr)
   for i in range(n_classes):
       mean_tpr += interp(all_fpr, fpr[i], tpr[i])

   # Finally average it and compute AUC
   mean_tpr /= n_classes

   fpr["macro"] = all_fpr
   tpr["macro"] = mean_tpr
   roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

   # Plot all ROC curves
   plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

   plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                  ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

   colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
   for i, color in zip(range(n_classes), colors):
       plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

   plt.plot([0, 1], [0, 1], 'k--', lw=lw)
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Some extension of Receiver operating characteristic to multi-class')
   plt.legend(loc=0) #pick best location
   plt.show()
   return

