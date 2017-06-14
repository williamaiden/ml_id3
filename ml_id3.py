# -*- coding: UTF-8 -*- 
'''
Created on 2017年6月14日

@author: William Aiden
'''

from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree

data = open(r'id3_data.csv','rb')
reader = csv.reader(data)
print(reader) #<_csv.reader object at 0x0000000005338648>
header = reader.next()
print(header) #['RID', 'age', 'income', 'student', 'credit_rating', 'class_buys_computer']

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1]) #no no yes yes yes no yes no 。。。
#     print(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[header[i]] = row[i]
#         print(header[i])
#         print(row[i])
    featureList.append(rowDict)

print(featureList)
# [{'credit_rating': 'fair', 'age': 'youth', 'student': 'no', 'income': 'high'}, 
# {'credit_rating': 'excellent', 'age': 'youth', 'student': 'no', 'income': 'high'}, 
# {'credit_rating': 'fair', 'age': 'middle_aged', 'student': 'no', 'income': 'high'}, 
# {'credit_rating': 'fair', 'age': 'senior', 'student': 'no', 'income': 'medium'}, 
# {'credit_rating': 'fair', 'age': 'senior', 'student': 'yes', 'income': 'low'}, 
# {'credit_rating': 'excellent', 'age': 'senior', 'student': 'yes', 'income': 'low'}, 
# {'credit_rating': 'excellent', 'age': 'middle_aged', 'student': 'yes', 'income': 'low'}, 
# {'credit_rating': 'fair', 'age': 'youth', 'student': 'no', 'income': 'medium'}, 
# {'credit_rating': 'fair', 'age': 'youth', 'student': 'yes', 'income': 'low'}, 
# {'credit_rating': 'fair', 'age': 'senior', 'student': 'yes', 'income': 'medium'}, 
# {'credit_rating': 'excellent', 'age': 'youth', 'student': 'yes', 'income': 'medium'}, 
# {'credit_rating': 'excellent', 'age': 'middle_aged', 'student': 'no', 'income': 'medium'}, 
# {'credit_rating': 'fair', 'age': 'middle_aged', 'student': 'yes', 'income': 'high'}, 
# {'credit_rating': 'excellent', 'age': 'senior', 'student': 'no', 'income': 'medium'}]

vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()
print(dummyX)
# [[ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
#  [ 0.  0.  1.  1.  0.  1.  0.  0.  1.  0.]
#  [ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]
#  [ 0.  1.  0.  0.  1.  0.  0.  1.  1.  0.]
#  [ 0.  1.  0.  0.  1.  0.  1.  0.  0.  1.]
#  [ 0.  1.  0.  1.  0.  0.  1.  0.  0.  1.]
#  [ 1.  0.  0.  1.  0.  0.  1.  0.  0.  1.]
#  [ 0.  0.  1.  0.  1.  0.  0.  1.  1.  0.]
#  [ 0.  0.  1.  0.  1.  0.  1.  0.  0.  1.]
#  [ 0.  1.  0.  0.  1.  0.  0.  1.  0.  1.]
#  [ 0.  0.  1.  1.  0.  0.  0.  1.  0.  1.]
#  [ 1.  0.  0.  1.  0.  0.  0.  1.  1.  0.]
#  [ 1.  0.  0.  0.  1.  1.  0.  0.  0.  1.]
#  [ 0.  1.  0.  1.  0.  0.  0.  1.  1.  0.]]
print(vec.get_feature_names())
# ['age=middle_aged', 'age=senior', 'age=youth', 
# 'credit_rating=excellent', 'credit_rating=fair', 
# 'income=high', 'income=low', 'income=medium', 
# 'student=no', 'student=yes']

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print(dummyY)
# [[0]
#  [0]
#  [1]
#  [1]
#  [1]
#  [0]
#  [1]
#  [0]
#  [1]
#  [1]
#  [1]
#  [1]
#  [1]
#  [0]]

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)
print(clf)
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')

with open("id3.dot",'w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

predictList = []
oneRowX = dummyX[0,:]
print(oneRowX)
# [ 0.  0.  1.  0.  1.  1.  0.  0.  1.  0.]
newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print(newRowX)
# [ 1.  0.  0.  0.  1.  1.  0.  0.  1.  0.]
predictList.append(newRowX)
predictedY = clf.predict(predictList)
print(predictedY)
# [1]
