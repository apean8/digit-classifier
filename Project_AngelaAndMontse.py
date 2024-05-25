
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Load the database
mat_file =  "BigDigits.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary

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

scoring = ['f1', 'accuracy']
hidden_layer = [1, 5, 10, 15, 20]
neigh = np.arange(1, 21)

times = []
f1_result = []
accuracy_result = []
conf_matrix_result = []
curve_result = []

for name, lws, clf in classifiers:
    print('\n Training %s' %name)

    if name == 'MLP':
        total_time = []
        mlp_scores = np.zeros((len(hidden_layer), len(hidden_layer), 3))

        for i, n_neurons1 in enumerate(hidden_layer):
            for j, n_neurons2 in enumerate(hidden_layer):
                clf.set_params(hidden_layer_sizes=(n_neurons1, n_neurons2))
                scores = cross_validate(clf, data_X, data_y, cv=5, scoring=scoring)
                total_time.append(scores['fit_time'].mean())

                media = scores['test_f1'].mean()
                mlp_scores[i, j, 0] = media
                mlp_scores[i, j, 1] = 1- media
                mlp_scores[i, j, 2] = scores['test_accuracy'].mean()
        
        min_index = np.unravel_index(np.argmin(mlp_scores[:, :, 1]), mlp_scores[:, :, 1].shape)
        optimal_i = min_index[0]
        optimal_j = min_index[1]

        optimal_neurons1 = hidden_layer[optimal_i]
        optimal_neurons2 = hidden_layer[optimal_j]
        print("Optimal configuration: ({}, {})".format(optimal_neurons1, optimal_neurons2))

        # To know the aprox. the training time of the classifier with two layers, we compute the mean of all
        times.append(np.mean(total_time))

        # We save the accuracy of the optimal value
        accuracy_result.append(mlp_scores[optimal_i, optimal_i, 2])

        # We compute the optimal value so we can estimate the predictions
        clf.set_params(hidden_layer_sizes=(optimal_neurons1, optimal_neurons2))
        pred_proba = cross_val_predict(clf, data_X, data_y, cv=5, method='predict_proba')
        pred = pred_proba[:,1]>0.5
        conf_matrix = confusion_matrix(data_y, pred, normalize='pred')
        fpr, tpr, thresholds = roc_curve(data_y, pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        curve_result.append([name, fpr, tpr, thresholds, roc_auc])
        conf_matrix_result.append(conf_matrix)

        plt.imshow(mlp_scores[:, :, 0])
        for i in range(len(hidden_layer)):
            for j in range(len(hidden_layer)):
                text = plt.text(j, i, "{:.2f}".format(mlp_scores[i, j, 0]),
                       ha="center", va="center", color="w")
        plt.colorbar()
        plt.xticks(np.arange(len(hidden_layer)), hidden_layer)
        plt.yticks(np.arange(len(hidden_layer)), hidden_layer)
        plt.xlabel('Neuronas en L2')
        plt.ylabel('Neuronas en L1')
        plt.show()

    elif name == 'KNN':
        knn_scores = []
        cv_errors = []
        total_time = []
        knn_accuracy = []

        for k in neigh:
            clf.set_params(n_neighbors=k)
            scores = cross_validate(clf, data_X, data_y, cv=5, scoring=scoring)
            total_time.append(scores['fit_time'].mean())

            media = scores['test_f1'].mean()
            knn_accuracy.append(scores['test_accuracy'].mean())
            knn_scores.append(media)
            cv_errors.append(1-media)
            

        plt.xticks(neigh)
        plt.xlabel('K-neighbors')
        plt.ylabel('F1-score')
        plt.plot(neigh, knn_scores)
        plt.show()

        # To know the aprox. the training time of the classifier with two layers, we compute the mean of all
        times.append(np.mean(total_time))

        # We compute the optimal value so we can estimate the predictions
        optimal = np.argmin(cv_errors)
        optimal_k = neigh[optimal]

        clf.set_params(n_neighbors=optimal_k)
        pred_proba = cross_val_predict(clf, data_X, data_y, cv=5, method='predict_proba')
        pred = pred_proba[:,1]>0.5
        conf_matrix = confusion_matrix(data_y, pred, normalize='pred')
        fpr, tpr, thresholds = roc_curve(data_y, pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        curve_result.append([name, fpr, tpr, thresholds, roc_auc])
        conf_matrix_result.append(conf_matrix)

        # We save the accuracy of the optimal value
        accuracy_result.append(knn_accuracy[optimal])

    else:
        scores = cross_validate(clf, data_X, data_y, cv=5, scoring=scoring)
        times.append(scores['fit_time'].mean())
        f1_result.append([name, scores['test_f1'], scores['test_f1'].mean()])
        accuracy_result.append(scores['test_accuracy'].mean())

        pred_proba = cross_val_predict(clf, data_X, data_y, cv=5, method='predict_proba')
        pred = pred_proba[:,1]>0.5
        conf_matrix = confusion_matrix(data_y, pred, normalize='pred')
        conf_matrix_result.append(conf_matrix)
        
        fpr, tpr, thresholds = roc_curve(data_y, pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        curve_result.append([name, fpr, tpr, thresholds, roc_auc])


# We compared the f1-scores of each classifier
for name, scores, media in f1_result:
    if name == 'Linear':
        c = 'blue'
    else:
        c = 'green'
    plt.plot(range(len(scores)), [media]*len(scores), '--', color=c)
    plt.plot(range(len(scores)), scores, '-o', label=name, color=c)
    plt.text(0, media + 0.001, '{:.4f}'.format(media), color=c)

plt.xlabel('Fold')
plt.ylabel('F1-score')
plt.legend()
plt.show()

# We compared the roc-curve and the AUC of each classifier
for name, fpr, tpr, thresholds, roc_auc in curve_result:
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# We compared the accuracy of each classifier
names = [clf[0] for clf in classifiers]
bar_container = plt.bar(names, accuracy_result)
plt.bar_label(bar_container, fmt='{:.4f}')
plt.show()

# We compared the times of training of each classifier
names = [clf[0] for clf in classifiers]
bar_container = plt.bar(names, times)
plt.ylabel('Time [s]')
plt.bar_label(bar_container, fmt='{:.4f}')
plt.show()

# We compared the confusion matrix of each classifier
for n, conf_matrix in enumerate(conf_matrix_result):
    plt.subplot(2, 2, n+1)
    plt.imshow(conf_matrix, cmap='magma')
    plt.colorbar()
    plt.title(names[n])
    plt.xticks(np.arange(conf_matrix.shape[1]))
    plt.yticks(np.arange(conf_matrix.shape[0]))

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if conf_matrix[i, j] < 0.5:
                text_color = 'white'
            else:
                text_color = 'black'
            plt.text(j, i, '{:.4f}'.format(conf_matrix[i, j]), ha='center', va='center', color=text_color)

plt.show(
