# This is for Programming Assignment 4 in CS 286
# reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

#Imports
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#inputs:
#y_test: numpy array with ground truth
#y_score: numpy array with scores predicted from a classifier
#n_classes: total number of output classes in the dataset 
#
#outputs: display the ROC plot
def displayROC(y_test, y_score, n_classes):
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

	# plot ROC
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='darkorange',
				lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()

