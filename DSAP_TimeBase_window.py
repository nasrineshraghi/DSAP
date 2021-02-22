
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:51:46 2020

@author: neshragh
- messing with adaptive threshold written by Nasrin which used to calculate 
    the threshold with each incoming point.
-This Version creates an adaptive threshold value at the end of each window
- Tested with the bubble data
"""


from sklearn.cluster import AffinityPropagation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import datetime
import os
import cProfile
import re
import time
import psutil
from sklearn import metrics
from scipy.spatial import distance
from skmultiflow.data.file_stream import FileStream
start_time = time.time()
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix
import cProfile
import re
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
#########%%%%%%%%%%%%%%%%%%%%%%%%%%%%########################################
# Creates landmark windows
def getwindowsize(start_time, time_period): 
    try:
        idx1 = np.where(df[:,2] >= start_time )[0][0] #first event, condition
        idx2 = np.where(df[:,2] >= start_time + time_period )[0][0] #
    except:
        idx1 = 0
        idx2 = 0
    return idx2 - idx1



def getAdaptiveThreshold(X,Y):
    epsilon = np.mean(distance.cdist(X,Y,'euclidean'))
    return epsilon

##Read the stream out of Sckit multiflow #####################################
#stream = FileStream("wifi.csv")
#df_dat = pd.read_csv('Wifi_test.csv')
#df = df_dat.to_numpy()
#df_dat = pd.read_csv('wifi_fake_bubble_test.csv')
#stream = FileStream("wifi_fake_bubble.csv")
#df = df_dat.to_numpy() 
csvName = 'ecounter_time.csv'
#csvName = 'wifi_fake_bubble_test.csv'
stream = FileStream(csvName)
df_dat = pd.read_csv(csvName)
df = df_dat.to_numpy()
Col1 = 1
Col2 = 0
timeColNum = 2




#windowsize = 100
# Window 1 onwards #######################################################################
ght = 1
X = []
while(ght<150):  
#while(stream.has_more_samples()):
###%%%# for window 1 only #############################################################################
    if ght ==1:        
        c = df[:,2].reshape((-1,1))
        start_time = c[0]
        time_period = 3600*20 #second
        windowsize = getwindowsize(start_time, time_period)
#        windowsize = 1000 
        X = stream.next_sample(windowsize)
        
        T = X[0]
        a = T[:,Col1].reshape((-1,1))
        b = T[:,Col2].reshape((-1,1))
        c = df[:,timeColNum].reshape((-1,1)) # todo: delete df
        ####################
        max_a = np.max(a)
        max_b = np.max(b)
        min_a = np.min(a)
        min_b = np.min(b)
        a = (a-min_a) / (max_a-min_a)
        b = (b - min_b) / (max_b-min_b)
        #################################
        X0 = np.concatenate((a,b), axis=1)
        # Compute Affinity Propagation for the first window
        af = AffinityPropagation(preference=-0, damping=.99, max_iter= 100 ).fit(X0)
        #af = AffinityPropagation(preference=-25, damping=.56, max_iter= 100 ).fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        my_centers = af.cluster_centers_
        n_clusters_ = len(cluster_centers_indices)


        #eps = getAdaptiveThreshold()
         # set threshold equal to half or mean the maximum euclidean distance
        #eps = np.max(distance.cdist(my_centers,X0,'euclidean'))/2
        eps = np.mean(distance.cdist(my_centers,X0,'euclidean'))
        print(eps)
        ght = ght +1
    else:
        
#        plt.figure(1)
#        plt.clf()
#        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#        for k, col in zip(range(n_clusters_), colors):
#            class_members = labels == k
#            cluster_center = X0[cluster_centers_indices[k]]
#            plt.plot(X0[class_members, 0], X0[class_members, 1], col+ '+', markersize=8)
#            plt.plot(cluster_center[0], cluster_center[1], 's', markerfacecolor=col,
#                 markeredgecolor='k', markersize=5)   
        
#        plt.plot(my_centers[:,0], my_centers[:,1], 'gs', markeredgecolor='k', markersize=5) #markerfacecolor=col, 
#        plt.plot(X[:, 0], X[:, 1],  '+', markersize=8)
# #############################################################################
        windowsize = getwindowsize(start_time, time_period)
        start_time = start_time + time_period
  
        if windowsize == 0:
            X = stream.next_sample(1)
            ght = ght +1
            continue
        X = stream.next_sample(windowsize)
        T = X[0]
        a = T[:,Col1].reshape((-1,1))
        b = T[:,Col2].reshape((-1,1))
        max_a = np.max(a)
        max_b = np.max(b)
        min_a = np.min(a)
        min_b = np.min(b)
        a = (a-min_a) / (max_a-min_a)
        b = (b - min_b) / (max_b-min_b)
        
        X = np.concatenate((a,b), axis=1) 
        
        
#         df
    #    X = X + np.random.normal(0,0.5,np.shape(X)) #Noise added
    #    X = X + np.sin(ght)+ np.random.normal(0,0.00025,np.shape(X)) # oscillating cluster location    
    #    X = X*2*np.sin(ght)                            # Changing cluster size,
    #    X = X*ght                            # Changing cluster size,
#        sft = ght*0.91 + np.random.normal(0,0.007,1) #concept-drift
#        X = X + sft  # random Shift added    
        
        plt.figure(1)
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels == k
            cluster_center = X0[cluster_centers_indices[k]]
            plt.plot(X0[class_members, 0], X0[class_members, 1], col+ '+', markersize=8)
            plt.plot(cluster_center[0], cluster_center[1], 's', markerfacecolor=col,
                     markeredgecolor='k', markersize=5)   
            
        plt.plot(my_centers[:,0], my_centers[:,1], 'gs', markeredgecolor='k', markersize=5) #markerfacecolor=col, 
        plt.plot(X[:, 0], X[:, 1],  '+', markersize=8)
        plt.pause(0.02)
        
        if np.size(X, axis =0) < windowsize: #count the x-->raws
            break
#        receiveddata = np.concatenate((receiveddata, X))
#        
#        D = distance_matrix(receiveddata, my_centers)
        outs = 0
        outpoint = []
        for i in range(windowsize):
            ed = [0]*n_clusters_   
    #        for kk in range(n_clusters_):    
    #            ed[kk] = distance.euclidean(my_centers[kk,:],X[i])
            nxpnt = X[i].reshape(1,2)
            ed = distance.cdist(my_centers,nxpnt,'euclidean')     
    #        idx = np.argmin(ed)    
    #        threshold = getAdaptiveThreshold(D, idx)
    #        print(threshold)
            if min(ed) > eps:
                outs = outs +1
                outpoint.append(X[i])
    #            print("TEST***************")
                
        if outs > 0:
            print(outs)
            Y=np.array(outpoint)        
            f = AffinityPropagation(preference=-0, damping=.99, max_iter= 100 ).fit(Y)
            out_centers = f.cluster_centers_
            #abel_out = f.labels_
    #        my_centers = out_centers
            my_centers = np.append(my_centers,out_centers,axis=0)
    #            cluster_center = X[cluster_centers_indices]
            plt.plot(Y[:, 0], Y[:, 1], 'r+', markersize=8)
            plt.plot(out_centers[:,0], out_centers[:,1], 'bs')
            rtb=1
            
            # set interim threshold equal to half the maximum euclidean distance of the reposditry points from the clusters
            int_eps = np.max(distance.cdist(out_centers,Y,'euclidean')) # Get interim threshold values from repository clusters
            print('int_eps',int_eps)
         ## Update cluster threshold size at the end of each window
            if eps < int_eps:
               eps = (int_eps + eps)/2  # If the new threshold is larger than the earlier one then increase the size
    #        else:
    #           eps = (int_eps + eps)/2 # If the new thresholds are smaller then this indicates 
                                   # that there might be smaller clusters away from the original clusters            
        ght = ght +1
        print(eps)
    
### Macro Cluster after AP restart    
#wtmic = # weghted macro
af_mac = AffinityPropagation(preference= -8, damping=.8, max_iter= 100 ).fit(my_centers)
cluster_centers_indices_mac = af_mac.cluster_centers_indices_
#labels_mac = af_mac.cluster_centers_
labelsm = af_mac.labels_
my_label = af_mac.predict(df[:,0:2])
##############################################################################
###############################################################################
n_cluster_ = len(cluster_centers_indices_mac)
print('Estimated number of macro clusters: %d' % n_cluster_)
n_cluste = len(my_centers)
print('Estimated number of micro clusters: %d' % n_cluste)
###############################################################################
#2D plot
#plt.close('all')
plt.figure(3)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
cluster_center = my_centers[cluster_centers_indices_mac]

#plt.scatter(df[:,0],df[:,1])

plt.scatter(my_centers[:,0], my_centers[:,1], linewidth=2,facecolors='none', s=100, edgecolor="silver")
plt.scatter(cluster_center[:,0],cluster_center[:,1], c='orangered',marker='+', s=1500, alpha=0.9)


#plt.plot(cluster_center[:,0],cluster_center[:,1], '+' , markerfacecolor='r', markersize=80 )


plt.title('DSAP Micro-Macro Clustering Result')
#plt.title('%d' % n_clusters_)


plt.xlabel('Position')
plt.ylabel('Count')

frame =plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)

#plt.ylim(0,20)

#plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
#           "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])

plt.show()


    
# #############################################################################
# =============================================================================
#print("Processing time: %s seconds" % (time.time() - start_time))
#
##Profiling and memory usage--------------------------------------------------
#process = psutil.Process(os.getpid())
#print("Memory Consumption:",process.memory_info().rss, 'bytes')   
#M= process.memory_info().rss /1000000
#print('(Or Megabyte:', M,')')
#
#cProfile.run('re.compile("foo|bar")')

# =============================================================================

    
#print('Silhouette Coefficient Micro:',metrics.silhouette_score(Y, abel_out, metric='euclidean'))
#print('Davies-Bouldin Index Micro:',davies_bouldin_score(Y, abel_out))
#print('Calinski-Harabasz Index Micro:',metrics.calinski_harabasz_score(Y, abel_out))
###############################################################################
#print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(my_centers, labelsm))  
#print('Silhouette Coefficient Macro:',metrics.silhouette_score(my_centers, labelsm, metric='euclidean'))
#print('Davies-Bouldin Index Macro:',davies_bouldin_score(my_centers, labelsm))
#cProfile.run('re.compile("foo|bar")')
###############################################################################


# #############################################################################
# Compute Affinity Propagation
#X = df[:3000,0:2]
#af = AffinityPropagation(preference=-23, damping=.96, max_iter= 100 ).fit(X)
#cluster_centers_indices = af.cluster_centers_indices_
#labels = af.labels_
#
#n_clusters_ = len(cluster_centers_indices)
#
#print('Estimated number of clusters: %d' % n_clusters_)
##print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
##print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
##print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
##print("Adjusted Rand Index: %0.3f"
##      % metrics.adjusted_rand_score(labels_true, labels))
##print("Adjusted Mutual Information: %0.3f"
##      % metrics.adjusted_mutual_info_score(labels_true, labels))
#'''print("Silhouette Coefficient: %0.3f"
#      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))'''
#
## #############################################################################
## Plot result
#
#plt.figure(4)
#plt.clf()
#
#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#for k, col in zip(range(n_clusters_), colors):
#    class_members = labels == k
#    cluster_center = X[cluster_centers_indices[k]]
#    plt.plot(X[class_members, 1], X[class_members, 0], col + '.')
#    plt.plot(cluster_center[1], cluster_center[0], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
##    for x in X[class_members]:
##        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
#
#plt.title('AP Clustering Algorthm, one week of Wifi data')
#plt.xlabel('SPACE ID')
#plt.ylabel('PHONE ID')
#plt.show()
################################################################