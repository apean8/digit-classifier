

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.matlab as matlab
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sympy import plot

# Load the database
mat_file =  "BigDigits.mat"
mat = matlab.loadmat(mat_file,squeeze_me=True) # dictionary
list(mat.keys()) # list vars

taska = True

data = mat["data"]      # read feature vectors
labs = mat["labs"] - 1  # read labels 1..10

allNlabs = np.unique(labs) # all labs 0 .. 9

classsiz = ()
for c in allNlabs:
    classsiz = classsiz + (np.size(np.nonzero(labs==c)),)  
print ('\n%% Class labels are: %s' % (allNlabs,))    
print ('%% Class frequencies are: %s' % (classsiz,))


# Let's say my digit is ...
myDigit = 1

otherDigits  = np.setdiff1d(allNlabs,myDigit)
other3Digits = np.random.permutation(otherDigits)[:9]

if taska:
    others = other3Digits
else:
    others = otherDigits

print ('class 1 = %s' % myDigit)
print ('class 2 = %s' % others)

# To construct a 2-class dataset you can use the same matrix
# data and change the vector of labels

aux = labs

positive = np.in1d(labs,myDigit)
negative = np.in1d(labs,others)
aux[positive] = 1  # class positive
aux[negative] = 0  # class negative

# Features
X = data[np.logical_or(positive,negative)]
# (unchanged) labels
y = aux[np.logical_or(positive,negative)]

# We divide the data into 5 folds. We use cross-validation
clf = LinearDiscriminantAnalysis()
scores = cross_val_score(clf, X, y, cv=5)
clf.fit(X, y)

pred = clf.predict(X)
proba = clf.predict_proba(X)

plt.plot(range(len(scores)), scores)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Cross-Validation Scores')
plt.show()

fpr, tpr, th = roc_curve(y, proba[:,1])
plt.plot(fpr, tpr)
plt.show()

clf1 = QuadraticDiscriminantAnalysis();
scores1 = cross_val_score(clf1, X, y, cv=5)
clf1.fit(X, y)

pred = clf1.predict(X)
proba = clf1.predict_proba(X)

plt.plot(range(len(scores1)), scores1)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Cross-Validation Scores')
plt.show()

fpr, tpr, th = roc_curve(y, proba[:,1])
plt.plot(fpr, tpr)
plt.show()

clf2 = MLPClassifier(max_iter=130, alpha=1e-4, solver='sgd',
tol=1e-4, random_state=1, learning_rate_init=.1)
scores2 = cross_val_score(clf2, X, y, cv=5)
clf2.fit(X, y)

pred = clf2.predict(X)
proba = clf2.predict_proba(X)

plt.plot(range(len(scores2)), scores2)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Cross-Validation Scores')
plt.show()

fpr, tpr, th = roc_curve(y, proba[:,1])
plt.plot(fpr, tpr)
plt.show()

clf3 = KNeighborsClassifier(n_neighbors=10)
scores3 = cross_val_score(clf3, X, y, cv=5)
clf3.fit(X, y)

pred = clf3.predict(X)
proba = clf3.predict_proba(X)

plt.plot(range(len(scores3)), scores3)
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Cross-Validation Scores')
plt.show()

fpr, tpr, th = roc_curve(y, proba[:,1])
plt.plot(fpr, tpr)
plt.show()

# Show some digits
hwmny = 20
negativeValues = np.random.permutation(np.where(y==0)[0])[:hwmny]
positiveValues = np.random.permutation(np.where(y==1)[0])[:hwmny]

img1 = np.reshape(X[positiveValues,:],(28*hwmny,28)).T
plt.figure(figsize=(10,3))
plt.imshow(img1, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Show digits in class positive (1) = '+str(myDigit) )
plt.show()


img2 = np.reshape(X[negativeValues,:],(28*hwmny,28)).T
plt.figure(figsize=(10,3))
plt.imshow(img2, cmap=plt.cm.gray_r)
plt.xticks([])
plt.yticks([])
plt.title('Show digits in class negative (0) = '+str(others) )
plt.show()






