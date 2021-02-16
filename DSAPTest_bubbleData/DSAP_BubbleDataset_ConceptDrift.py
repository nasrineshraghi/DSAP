
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:51:46 2020

@author: neshragh
Swadesh messing with adaptive threshold
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:52:54 2020
Final version of DSAP for e-counter dataset
@author: neshragh
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

##Read the stream out of Sckit multiflow #####################################
#stream = FileStream("wifi.csv")
#
#
#df_dat = pd.read_csv('Wifi_test.csv')
#df = df_dat.to_numpy() 


df_dat = pd.read_csv('wifi_fake_bubble_test.csv')
stream = FileStream("wifi_fake_bubble.csv")
df = df_dat.to_numpy() 

##############################################################################
##open a dataset
#dc = pd.read_csv('week4event.csv')
##df_data = pd.read_csv('week1.csv')
#df = dc.loc[(dc.Date == '3/18/2019') & (dc.Time >= '10:00:00') & (dc.Time <= '11:00:00')]
##df = dc.loc[(dc.Date >= '4/15/2019') & (dc.Date <= '4/29/2019')] 
# #############################################################################
X = stream.next_sample(1000)
a = X[0].reshape((-1,1))
b = X[1].reshape((-1,1))
c = df[:,2].reshape((-1,1))
start_time = c[999]
time_period = 3600 #second
X0 = np.concatenate((a,b), axis=1)
#X = np.asarray(X)
#X = X[0:200,0:2]

#Choose data for algorithm
#X= df_dat[:200]
#plt.scatter(X[:,0],X[:,1])
#plt.show()

#labels_true = df.loc[df.index<90000,'Time'].to_numpy()



# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-8, damping=.95, max_iter= 100 ).fit(X0)
#af = AffinityPropagation(preference=-25, damping=.56, max_iter= 100 ).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_
my_centers = af.cluster_centers_
n_clusters_ = len(cluster_centers_indices)

# #############################################################################
# Plot result
plt.close('all')
plt.figure(1)
plt.clf()
# =============================================================================
#plt.scatter(X[:,0],X[:,1], color='c',alpha=0.3,  linewidth=4)
# =============================================================================
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X0[cluster_centers_indices[k]]
    plt.plot(X0[class_members, 0], X0[class_members, 1], col + '+', markersize=8)
    plt.plot(cluster_center[0], cluster_center[1], 's', markerfacecolor=col,
             markeredgecolor='k', markersize=15)
    F=X0[class_members, 0]

  
outs = 0
outpoint = []

def getwindowsize(start_time, time_period):
    try:
        idx1 = np.where(df[:,2] >= start_time )[0][0] #first event, condition
        idx2 = np.where(df[:,2] >= start_time + time_period )[0][0] #
    except:
        idx1 = 0
        idx2 = 0
    return idx2 - idx1

def getAdaptiveThreshold(D, idx): # idx is the index of closest center , D distance matrix of all data points and centers in window
#        print(np.shape(X))
#        print(np.shape(C))
        cluster_idx = np.argmin(D, axis = 1) # index of the min in each row which blongs to cluster center
#       print(np.shape(D))
        DD = np.min(D, axis = 1) # value of the min in each row which blongs to cluster center
        T = np.max(DD[cluster_idx == idx])
        return T

#datapoint until now in the window
receiveddata = stream.next_sample(1)
a = receiveddata[0].reshape((-1,1))
b = receiveddata[1].reshape((-1,1))
receiveddata = np.concatenate((a,b), axis=1)
#windowsize = 100
ght = 1
X = []
while(ght<25):  
#while(stream.has_more_samples()):
    windowsize = getwindowsize(start_time, time_period)
    start_time = start_time + time_period
    windowsize = 400    
    if windowsize == 0:
        X = stream.next_sample(1)
        continue
    X = stream.next_sample(windowsize)
    a = X[0].reshape((-1,1))
    b = X[1].reshape((-1,1))
    X = np.concatenate((a,b), axis=1) 
    X = X + np.random.normal(0,0.25,np.shape(X)) #Noise added
    sft = ght*0.91 + np.random.normal(0,0.007,1) #concept-drift
    X = X + sft  # random Shift added
    
    plt.figure(1)
    plt.clf()   
    
    
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X0[cluster_centers_indices[k]]
        plt.plot(X0[class_members, 0], X0[class_members, 1], col+ '+', markersize=8)
        plt.plot(cluster_center[0], cluster_center[1], 's', markerfacecolor=col,
                 markeredgecolor='k', markersize=5)   
        
    plt.plot(my_centers[:,0], my_centers[:,1], 'gs', markeredgecolor='k', markersize=5) #markerfacecolor=col, 
    plt.plot(X[:, 0], X[:, 1],  '+', markersize=8)
    plt.pause(0.1)
    
    if np.size(X, axis =0) < windowsize: #count the x-->raws
        break
    receiveddata = np.concatenate((receiveddata, X))
    
    D = distance_matrix(receiveddata, my_centers)
    outs = 0
    outpoint = []
    for i in range(windowsize):
        ed = [0]*n_clusters_   
        for kk in range(n_clusters_):            
#            class_members = labels == kk
#            cluster_center = X[cluster_centers_indices[kk]]
#            print('cluster cent',cluster_center)
#            print(str(i)+ ' '+ str(kk)+ ' '+ str(n_clusters_))
            ed[kk] = distance.euclidean(my_centers[kk,:],X[i])
#            print(ed[k])
            
        idx = np.argmin(ed)    
        threshold = getAdaptiveThreshold(D, idx)
#        print(threshold)
        if min(ed) > threshold:
            outs = outs +1
            outpoint.append(X[i])
#            print("TEST***************")

    if outs > 0:
        Y=np.array(outpoint)        
        f = AffinityPropagation(preference=-8, damping=.95, max_iter= 100 ).fit(Y)
        out_centers = f.cluster_centers_
        #abel_out = f.labels_
#        my_centers = out_centers
        my_centers = np.append(my_centers,out_centers,axis=0)
#            cluster_center = X[cluster_centers_indices]
        plt.plot(Y[:, 0], Y[:, 1], 'r+', markersize=8)
        plt.plot(out_centers[:,0], out_centers[:,1], 'bs')
        rtb=1
#        plt.close('all')
#            Y = np.concatenate((Y,my_centers))
#        else:
#            Y= np.array(cluster_center)
            
# =============================================================================
#             plt.close('all')
#             plt.figure(1)
#             # ===============================================================
#             plt.scatter(X[:,0],X[:,1], color='c',alpha=0.3,  linewidth=4)
#             # ===============================================================
#             colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#             
#             for k, col in zip(range(n_clusters_), colors):
#                 class_members = labels == k
#                 cluster_center = X[cluster_centers_indices[k]]
#                 plt.plot(X[class_members, 0], X[class_members, 1], col + '+', markersize=8)
#                 plt.plot(cluster_center[0], cluster_center[1], 's', markerfacecolor=col,
#                          markeredgecolor='k', markersize=15)
# =============================================================================
            ###############################
#            plt.clf()
            #plt.scatter(Y[:, 0], Y[:, 1], color='m', alpha=0.1, linewidth=2)
#            Y= outpoint[0:100,:]
            
            
           
            #print('Estimated number of clusters: %d' % n_cluster_)
#            n_clus_ = len(cluster_centers_indice)
#            for p, pol in zip(range(n_clus_), colors):
#                class_member = label == p
#                cluster_cente = Y[cluster_centers_indice[p]]
##                plt.plot(Y[class_member, 0], Y[class_member, 1], pol + '+', markersize=8)                
#                plt.plot(cluster_cente[0], cluster_cente[1], 'h', markerfacecolor=pol,
##                         markeredgecolor='k', markersize=15)                
#                F= Y[class_member, 0]
#            plt.show()
                        

            ### Macro Cluster after AP restart
    ght = ght +1
    print(threshold)
                    
#print('Exemplars:',X[cluster_centers_indices])
#print('Outliers',Y[cluster_centers_indice])            

#concat two arrays of outliers and examplar to generate macro clusters
#eX= X[cluster_centers_indices]
#eY= Y[cluster_centers_indice]
#macro = np.concatenate((eX,eY), axis=0)




af_mac = AffinityPropagation(preference= -1374, damping=.55, max_iter= 100 ).fit(my_centers)
cluster_centers_indices_mac = af_mac.cluster_centers_indices_
#labels_mac = af_mac.cluster_centers_
labelsm = af_mac.labels_
my_label = af_mac.predict(df[:,0:2])
#n_clustersmac_ = len(cluster_centers_indices_mac)
#for m, colm in zip(range(n_clustersmac_), colors):
#    class_members_mac = labels_mac == m
#    cluster_center_mac = macro[cluster_centers_indices_mac[m]]
#    plt.plot(macro[class_members_mac, 0], macro[class_members_mac, 1], col + '+', markersize=8)
#    plt.plot(cluster_center_mac[0], cluster_center_mac[1], 'P', markerfacecolor=col,
#             markeredgecolor='k', markersize=15)
#    F=macro[class_members_mac, 0]
##############################################################################
#3D plot    
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
#    
#ax.view_init(elev=20., azim=45)
#    
#
#ax.scatter(df[:,0], df[:,1], df[:,2], c=my_label, marker="o", picker=True)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Floor')
#
###############################################################################
#n_cluster_ = len(cluster_centers_indices_mac)
#print('Estimated number of macro clusters: %d' % n_cluster_)
#n_cluste = len(my_centers)
#print('Estimated number of micro clusters: %d' % n_cluste)
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


plt.title('DSAP Micro-Macro Clustering Result:Wifi dataset')
#plt.title('%d' % n_clusters_)


plt.xlabel('SPACEID')
plt.ylabel('PHONEID')

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
