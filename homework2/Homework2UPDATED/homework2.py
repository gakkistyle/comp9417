import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

#read the data from dataset
df = pd.read_csv('titanic.csv')
target_name = "Survived"
target = df[target_name].values.reshape(-1,1)
all_features = df[0:].values[:,0:5]

#normalization
all_features_nom = (all_features-np.min(all_features,axis=0))/(np.max(all_features,axis=0)-np.min(all_features,axis=0))

#for partA to train the model and get the accuracy scores
cls1 = DecisionTreeClassifier()
cls1.fit(all_features_nom[0:620],target[0:620])

print('Accuracy score for training set: ',cls1.score(all_features_nom[0:620],target[0:620]))
print('Accuracy score for test set: ',cls1.score(all_features_nom[620:],target[620:]))

#preparation of getting min_samples_leaf and the plot
min_samples_leaf = 2
optimal_auc_test = 0
plt.xticks(range(2,21))
auc_trainset = []
auc_testset = []
for i in range(2,21):
    cls2 = DecisionTreeClassifier(min_samples_leaf = i)
    cls2.fit(all_features_nom[0:620],target[0:620])
    auc_train = roc_auc_score(target[0:620],cls2.predict_proba(all_features_nom[0:620])[:620,1])
    auc_test =  roc_auc_score(target[620:],cls2.predict_proba(all_features_nom[620:])[:620,1])
    auc_trainset.append(auc_train)
    auc_testset.append(auc_test)
    if auc_test > optimal_auc_test:
        optimal_auc_test = auc_test
        min_samples_leaf = i


#show the answers
print("optimal number of min_samples_leaf is : ",min_samples_leaf)
plt.plot(range(2,21),auc_trainset,'-o',c='red',label ='Auc for training data' )
plt.plot(range(2,21),auc_testset,'-o',c='blue',label='Auc for test data')
plt.legend()
plt.show()


#probability for P(S=true|G=female,C=1)
#count for people having that condition and count for people surviving under that condition
count_for_condition = 0
count_for_s = 0
for p in range(len(all_features)):
    if all_features[p][0]==1 and all_features[p][1] == 1:
        count_for_condition += 1
        if target[p][0] == 1:
            count_for_s += 1

pro = count_for_s/count_for_condition
print ('The possibility is:',pro)
