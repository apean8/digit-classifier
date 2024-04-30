
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab
from sklearn.calibration import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Load the database
mat_file =  "BigDigits.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

taska = True

data_X = mat["data"]      # read feature vectors
labs = mat["labs"] - 1  # read labels 1..10

allNlabs = np.unique(labs) # all labs 0 .. 9

myDigit = 1

others = np.setdiff1d(allNlabs,myDigit)
print ('Positive class = %s' % myDigit)
print ('Negative class = %s' % others)

data_y = np.in1d(labs,myDigit)

# We create a matrix with all the classifiers we're going to used
classifiers = [
    ("Linear", 1, LinearDiscriminantAnalysis()),
    ("Quadratic", 1, QuadraticDiscriminantAnalysis()),
    ("MLP",1, MLPClassifier(max_iter=1000)),
    ("KNN", 1, KNeighborsClassifier())
]

score_result = []
curve_result = []
hidden_layer = [1, 5, 10, 15, 20]
neigh = np.arange(1, 21)

for name, lws, clf in classifiers:
    print("\n  Training %s" % name)

    if name == 'MLP':
        mlp_scores = np.zeros((len(hidden_layer), len(hidden_layer)))
        mlp_curve = []
        for i, n_neurons1 in enumerate(hidden_layer):
            for j, n_neurons2 in enumerate(hidden_layer):
                clf.set_params(hidden_layer_sizes=(n_neurons1, n_neurons2))
                scores = cross_val_score(clf, data_X, data_y, cv=5, scoring='f1')
                pred = cross_val_predict(clf, data_X, data_y, cv=5, method='predict_proba')
                fpr, tpr, thresholds = roc_curve(data_y, pred[:, 1])
                roc_auc = auc(fpr, tpr)
                media = np.mean(scores)

                mlp_scores[i, j] = media
                mlp_curve.append([fpr, tpr, thresholds, roc_auc, n_neurons1, n_neurons2])
        
        best= np.argmax([item[3] for item in mlp_curve])
        print(best)
        curve_result.append([name, mlp_curve[best][0], mlp_curve[best][1], mlp_curve[best][2], mlp_curve[best][3]])
        plt.imshow(mlp_scores)
        for i in range(len(hidden_layer)):
            for j in range(len(hidden_layer)):
                text = plt.text(j, i, "{:.2f}".format(mlp_scores[i, j]),
                       ha="center", va="center", color="w")
        plt.colorbar()
        plt.xticks(np.arange(len(hidden_layer)), hidden_layer)
        plt.yticks(np.arange(len(hidden_layer)), hidden_layer)
        plt.xlabel('Neuronas en L2')
        plt.ylabel('Neuronas en L1')
        plt.show()

    if name == 'KNN':
        knn_scores = []
        knn_curve = []
        for k in neigh:
            clf.set_params(n_neighbors=k)
            scores = cross_val_score(clf, data_X, data_y, cv=5, scoring='f1')
            pred = cross_val_predict(clf, data_X, data_y, cv=5, method='predict_proba')
            fpr, tpr, thresholds = roc_curve(data_y, pred[:, 1])
            roc_auc = auc(fpr, tpr)
            media = np.mean(scores)
            knn_scores.append(media)

            knn_curve.append([fpr, tpr, thresholds, roc_auc])
        
        plt.xticks(neigh)
        plt.xlabel('K-neighbors')
        plt.ylabel('F1-score')
        plt.plot(neigh, knn_scores)
        plt.show()

        best = np.argmax([item[3] for item in knn_curve])
        best_k = neigh[best]
        curve_result.append([name, knn_curve[best][0], knn_curve[best][1], knn_curve[best][2], knn_curve[best][3]])
        for fpr, tpr, thresholds, roc_auc in knn_curve:
            plt.plot(fpr, tpr, color='lemonchiffon')
        plt.plot(knn_curve[best][0], knn_curve[best][1], color='orange', label='Best ROC Curve (K={})'.format(best_k))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

    else:
        scores = cross_val_score(clf, data_X, data_y, cv=5, scoring='f1')
        pred = cross_val_predict(clf, data_X, data_y, cv=5, method='predict_proba')
        fpr, tpr, thresholds = roc_curve(data_y, pred[:, 1])
        roc_auc = auc(fpr, tpr)
        media = np.mean(scores)

        score_result.append([name, scores, media])
        curve_result.append([name, fpr, tpr, thresholds, roc_auc])

for name, scores, media in score_result:
    if name == 'Linear':
        c = 'blue'
    else:
        c = 'green'
    plt.plot(range(len(scores)), [media]*len(scores), '--', color=c)
    plt.plot(range(len(scores)), scores, '-o', label=name, color=c)
    plt.text(0, media + 0.001, "{:.4f}".format(media), color=c)

plt.xlabel('Fold')
plt.ylabel('F1-score')
plt.legend()
plt.show()

for name, fpr, tpr, thresholds, roc_auc in curve_result:
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()