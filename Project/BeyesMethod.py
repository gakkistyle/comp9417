import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from pattern.en import lemma

from sklearn.metrics import accuracy_score, classification_report

# make the label code manually
def convert(text):
    result = []
    for c in text:
        if c == 'ARTS CULTURE ENTERTAINMENT':
            result.append(0)
        elif c == 'BIOGRAPHIES PERSONALITIES PEOPLE':
            result.append(1)
        elif c == 'DEFENCE':
            result.append(2)
        elif c == 'DOMESTIC MARKETS':
            result.append(3)
        elif c == 'FOREX MARKETS':
            result.append(4)
        elif c == 'HEALTH':
            result.append(5)
        elif c == 'MONEY MARKETS':
            result.append(6)
        elif c == 'SCIENCE AND TECHNOLOGY':
            result.append(7)
        elif c == 'SHARE LISTINGS':
            result.append(8)
        elif c == 'SPORTS':
            result.append(9)
        elif c == 'IRRELEVANT':
            result.append(10)
    return result

def converse_convert(n):
    if n == 0:
        return 'ARTS CULTURE ENTERTAINMENT'
    elif n == 1:
        return 'BIOGRAPHIES PERSONALITIES PEOPLE'
    elif n == 2:
        return 'DEFENCE'
    elif n == 3:
        return 'DOMESTIC MARKETS'
    elif n == 4:
        return 'FOREX MARKETS'
    elif n == 5:
        return 'HEALTH'
    elif n == 6:
        return 'MONEY MARKETS'
    elif n == 7:
        return 'SCIENCE AND TECHNOLOGY'
    elif n == 8:
        return 'SHARE LISTINGS'
    elif n == 9:
        return 'SPORTS'
    elif n == 10:
        return 'IRRELEVANT'


#lemmatization
def lemmatization(word_set):
    words = []
    for i in range(len(word_set)):
        words.append(word_set[i].split(','))
    for i in range(len(words)):
        for j in range(len(words[i])):
            words[i][j] = lemma(words[i][j])
    for i in range(len(words)):
        words[i] = ','.join(words[i])
    return words

# function that makes suggestion article for each topic
def create_suggestion(predicted_result,predict_proba_result,i):
    suggestion_for_i = np.where(predicted_result == i) + np.array([9501])
    sorted_num_plus_predic_y2 = predict_proba_result[np.argsort(predict_proba_result[:, i + 1])]
    suggestion = []
    if len(suggestion_for_i[0]) < 5:
        print('suugestion for',converse_convert(i),'is:',sorted_num_plus_predic_y2[-1:-6:-1,0])
        suggestion.append(sorted_num_plus_predic_y2[-1:-6:-1,0])
    elif len(suggestion_for_i[0]) <= 7:
        print('suggestion for', converse_convert(i), 'is:',sorted_num_plus_predic_y2[-1:-len(suggestion_for_i[0])-1:-1,0])
        suggestion.append(sorted_num_plus_predic_y2[-1:-len(suggestion_for_i[0])-1:-1,0])
    else:
        print('suggestion for', converse_convert(i), 'is:', sorted_num_plus_predic_y2[-1:-11:-1, 0])
        suggestion.append(sorted_num_plus_predic_y2[-1:-11:-1, 0])
    return suggestion[0]

# function that calculate the precision recall and f1 for each recommendation
def performance_measure(target_set,prediction_set):
    num_hit = len(set(prediction_set).intersection(set(target_set)))
    precision = float(num_hit/len(prediction_set))
    recall = float(num_hit/len(target_set))
    if precision+recall != 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = 0

    return precision,recall,f1


#train the training set
df = pd.read_csv('training.csv')
words_train = lemmatization(df['article_words'].values)

#compare feature numbers before and after lemmatization
tf = TfidfVectorizer()
tf_o = TfidfVectorizer()
bag_of_words = tf.fit_transform(words_train)
bag_of_words_original = tf_o.fit_transform(df['article_words'].values)
print('feature number before lemmatization:',len(tf_o.get_feature_names()))
print('feature number after lemmatization:',len(tf.get_feature_names()))
X = bag_of_words

# label code
y = df['topic'].values
y = convert(y)

# do cross validation to find best alpha,the scope is from 0.0 to 10.0,step = 0.1
MB = MultinomialNB()
scope = np.arange(0.0,10.0,0.01).tolist()
parameters = {'alpha':scope}
clf = GridSearchCV(MB, parameters,cv=10,scoring='f1_macro')
model = clf.fit(X,y)
print('Best alpha:',model.best_params_)


#calculate accuracy for the train and test set.
df2 = pd.read_csv('test.csv')
words_test = lemmatization(df2['article_words'].values)
bag_of_words_2 = tf.transform(words_test)
#bag_of_words_2 = tf.transform(df2['article_words'].values)
X2 = bag_of_words_2
y2 = df2['topic'].values
y2 = convert(y2)
predicted_y1 = model.predict(X)
predicted_y2 = model.predict(X2)
predicted_pro_y2 = model.predict_proba(X2)

print('Accuracy rate for the training:',accuracy_score(y,predicted_y1))
print('Accuracy rate for the test:',accuracy_score(y2, predicted_y2))
print('f1-macro for the training:',model.best_score_)
num_plus_predic_y2 = np.hstack((df2['article_number'].values.reshape(-1,1),predicted_pro_y2))

# do the suggestion for each topic
suggestion_set = []
for i in range(10):
    suggestion = create_suggestion(predicted_y2,num_plus_predic_y2,i)
    suggestion_set.append(suggestion)


# Precision  Recall F1 for the whole test set
print(classification_report(y2, predicted_y2))


#make performance measurement for each topic recommendation
for i in range(10):
    real_for_i = (np.array([9501])+np.where(np.array(y2)[:]==i))[0]
    precision,recall,f1 = performance_measure(real_for_i,suggestion_set[i])
    print('')
    print('Precision for',converse_convert(i),'recommendation set is',precision)
    print('Recall for',converse_convert(i),'recommendation set is',recall)
    print('F1 score for',converse_convert(i),'recommendation set is',f1)

