#!/usr/bin/env python2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
###3333
iris = load_iris()
X = iris.data
Y = iris.target
####33
#CLASE 1
c1 = Y==0
xc1=iris.data[0:50,0:2]
xc1=np.asmatrix(xc1)
xc2=iris.data[50:100,0:2]
xc3=iris.data[100:150,0:2]
#xc1
print('xc1')
xc1_shape = np.shape(xc1)
xc1_rows = xc1_shape[0]
#xc1_70
xc1_train_size = np.around(xc1_rows*.7)
xc1_test_size = xc1_rows - xc1_train_size
#xc1_70
xc1_train_set_index = np.random.choice(int(xc1_rows),int(xc1_train_size), replace = False)
xc1_train_set_index = np.sort(xc1_train_set_index)
xc1_train_set = np.sort(xc1_train_set_index)
xc1_train_set = xc1[xc1_train_set_index,:]
xc1_all_index = np.arange(0,xc1_rows)
print('all index')
print(xc1_all_index)
print('70% of index')
print(xc1_train_set_index)
#xc1_30
xc1_test_set_index = np.setxor1d(xc1_train_set_index,xc1_all_index)
print('30% of index')
print(xc1_test_set_index)
xc1_test_set = xc1[xc1_test_set_index,:]
#xc2
print('xc2')
xc2_shape = np.shape(xc2)
xc2_rows = xc2_shape[0]
xc2_train_size = np.around(xc2_rows*.7)
xc2_test_size = xc2_rows - xc2_train_size
#xc2_70
xc2_train_set_index = np.random.choice(int(xc2_rows),int(xc2_train_size), replace = False)
xc2_train_set_index = np.sort(xc2_train_set_index)
xc2_train_set = np.sort(xc2_train_set_index)
xc2_train_set = xc2[xc2_train_set_index,:]
xc2_all_index = np.arange(0,xc2_rows)
print('all index')
print(xc2_all_index)
print('70% of index')
print(xc2_train_set_index)
#xc2_30
xc2_test_set_index = np.setxor1d(xc2_train_set_index,xc2_all_index)
print('30% of index')
print(xc2_test_set_index)
xc2_test_set = xc2[xc2_test_set_index,:]
#xc3
print('xc3')
xc3_shape = np.shape(xc3)
xc3_rows = xc3_shape[0]
xc3_train_size = np.around(xc3_rows*.7)
xc3_test_size = xc3_rows - xc3_train_size
#xc3_70
xc3_train_set_index = np.random.choice(int(xc3_rows),int(xc3_train_size), replace = False)
xc3_train_set_index = np.sort(xc3_train_set_index)
xc3_train_set = np.sort(xc3_train_set_index)
xc3_train_set = xc3[xc3_train_set_index,:]
xc3_all_index = np.arange(0,xc3_rows)
print('all index')
print(xc3_all_index)
print('70% of index')
print(xc3_train_set_index)
#xc3_30
xc3_test_set_index = np.setxor1d(xc3_train_set_index,xc3_all_index)
print('30% of index')
print(xc3_test_set_index)
xc3_test_set = xc2[xc3_test_set_index,:]
#####3 tags
yc1_70=iris.target[0:50]
yc2_70=iris.target[50:100]
yc3_70=iris.target[100:150]
##3
yc1_30=iris.target[0:50]
yc2_30=iris.target[50:100]
yc3_30=iris.target[100:150]

######33
xtrain=np.concatenate((xc1_train_set_index, xc2_train_set_index, xc3_train_set_index), axis=0)
#xtrain=xtrain.reshape(xtrain.size,1)
print('xtrain')
print(xtrain)
ytrain=np.concatenate((yc1_70, yc2_70,yc3_70), axis=0)
print('ytrain')
print(ytrain)
###
xtest=np.concatenate((xc1_test_set_index,xc2_test_set_index,xc3_test_set_index), axis=0)
print('ytrain')
print(xtest)
ytest=np.concatenate((yc1_30, yc2_30,yc3_30), axis=0)
print('ytest')
print(ytest)
#####33
k=3
print('centroids')
#centroids=xc1_train_set_index[k:,]
centroids=np.random.choice(xtrain, k)
print(centroids)
####33
print('xtrain.shape')
print(xtrain.shape)
xtrain_num_rows=len(xtrain)
#print(xtrain_num_rows)
square_distance=np.zeros((xtrain_num_rows, k))
#square_distance=np.zeros((centroids, k))
print('square_distance shape')
print(np.shape(square_distance))
#######33
#print(np.shape(xtrain))
print('centroids, shape')
print(np.shape(centroids))
plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], s=40,linewidths=5)
plt.title('antes')
#plt.show()
colors = 10*["g","r","c","b","k"]
####333
class k_means:
    def __init__(self, k=3, tol=0.001, max_iter=10):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        ######3
    def fit(self,data):
        #####33
        self.centroids ={}
        #####33
        for i in range(self.k):
            self.centroids[i] = data[i]
            #####33
        for i in range(self.max_iter):
            self.classifications = {}
            #####33
            for i in range(self.k):
                self.classifications[i] = []
                ####3
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids = dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
                ####333
            optimized = True
            #####3
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
                    ####333
            if optimized:
                break
                ####333
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
        ##3333
clf = k_means()
clf.fit(X)
###333
##3333
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
	plt.subplot(1,2,2)
	plt.scatter(featureset[0], featureset[1], color=color, s=40, linewidths=5)
#33333
for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=40, linewidths=5)

##333
plt.title('despues')
plt.show()
