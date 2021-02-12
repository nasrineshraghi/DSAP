#DSAP with time window --> count-based
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


df_dat = pd.read_csv('wifi.csv')
stream = FileStream("wifi.csv")

##################################################################
##open a dataset
#dc = pd.read_csv('week4event.csv')
##df_data = pd.read_csv('week1.csv')
#df = dc.loc[(dc.Date == '3/18/2019') & (dc.Time >= '10:00:00') & (dc.Time <= '11:00:00')]
##df = dc.loc[(dc.Date >= '4/15/2019') & (dc.Date <= '4/29/2019')] 
# #############################################################################
X = stream.next_sample(200)
a = X[0].reshape((-1,1))
b = X[1].reshape((-1,1))
X = np.concatenate((a,b), axis=1)
#X = np.asarray(X)
#X = X[0:200,0:2]
df = df_dat.to_numpy() 
#Choose data for algorithm
#X= df_dat[:200]
#plt.scatter(X[:,0],X[:,1])
#plt.show()

#labels_true = df.loc[df.index<90000,'Time'].to_numpy()



# #############################################################################
# Compute Affinity Propagation
af = AffinityPropagation(preference=-8, damping=.88, max_iter= 100 ).fit(X)
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
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '+', markersize=8)
    plt.plot(cluster_center[0], cluster_center[1], 's', markerfacecolor=col,
             markeredgecolor='k', markersize=15)
    F=X[class_members, 0]
    
outs = 0
outpoint = []

windowsize = 200
while(stream.has_more_samples()):
    X = stream.next_sample(windowsize)
    a = X[0].reshape((-1,1))
    b = X[1].reshape((-1,1))
    X = np.concatenate((a,b), axis=1)
    if np.size(X, axis =0) < windowsize:
        break
    for i in range(windowsize):
        ed = [0]*n_clusters_   
        for kk in range(n_clusters_):            
#            class_members = labels == kk
#            cluster_center = X[cluster_centers_indices[kk]]
#            print('cluster cent',cluster_center)
#            print(str(i)+ ' '+ str(kk)+ ' '+ str(n_clusters_))
            ed[kk] = distance.euclidean(my_centers[kk,:],X[i])
#            print(ed[k])
            
        if min(ed) >2.0:
            outs = outs +1
            outpoint.append(X[i])
            
        if outs >= 80:   
            Y=np.array(outpoint)
#            cluster_center = X[cluster_centers_indices]
           
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
            f = AffinityPropagation(preference=-1, damping=.93, max_iter= 100 ).fit(Y)
            abel_out = f.labels_
            
#            print(i,Y.shape)
            Y =[]
            n_cluster_y = f.cluster_centers_indices_
            my_centers_out = f.cluster_centers_
            
            n_clusters_out = np.size(my_centers, 0)
            
            n_cluster_out = len(n_cluster_y)
            
            #compare outpoints with cluster centers
            comp =0
            al = []
            ed_out = [0]*n_clusters_
            for p in range(n_cluster_out):
                for out in range(n_clusters_):
                    ed_out[out] = distance.euclidean(my_centers[out,:],my_centers_out[p,0:2])
                    if min(ed_out) > 3.0:
#                        comp = comp +1
                        hgt = my_centers_out[p,0:2].reshape(1,2)
                        al = np.concatenate((hgt,my_centers))
                        
                        #distribution detection
#                    else:
#                        fout = AffinityPropagation(preference=-1, damping=.93, max_iter= 100 ).fit(al)
#                        abel_out = fout.labels_
            if len(al)>0:
                my_centers = al                   
            
            
           
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
                        
            outs = 0
            outpoint = []
               
            ### Macro Cluster after AP restart
            
                    
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
n_cluster_ = len(cluster_centers_indices_mac)
print('Estimated number of macro clusters: %d' % n_cluster_)
n_cluste = len(my_centers)
print('Estimated number of micro clusters: %d' % n_cluste)
###############################################################################
#2D plot
plt.close('all')
plt.figure(2)
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
print("Processing time: %s seconds" % (time.time() - start_time))

#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')


# =============================================================================

    
#print('Silhouette Coefficient Micro:',metrics.silhouette_score(Y, abel_out, metric='euclidean'))
#print('Davies-Bouldin Index Micro:',davies_bouldin_score(Y, abel_out))
#print('Calinski-Harabasz Index Micro:',metrics.calinski_harabasz_score(Y, abel_out))
###############################################################################
print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(my_centers, labelsm))  
print('Silhouette Coefficient Macro:',metrics.silhouette_score(my_centers, labelsm, metric='euclidean'))
print('Davies-Bouldin Index Macro:',davies_bouldin_score(my_centers, labelsm))
cProfile.run('re.compile("foo|bar")')
