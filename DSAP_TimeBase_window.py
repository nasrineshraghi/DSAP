
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
start_time_Org = time.time()
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix
import matplotlib
matplotlib.use('Qt5Agg')


###################  Function defenitions #####################################
# Return the number of points for each window 
def getwindowsize(start_time, time_period): 
    try:
        idx1 = np.where(df[:,2] >= start_time )[0][0] #first event, condition
        idx2 = np.where(df[:,2] >= start_time + time_period )[0][0] #
    except:
        idx1 = 0
        idx2 = 0
    return idx2 - idx1


#return the average Euc distance of points
def getAdaptiveThreshold(X,Y):
    epsilon = np.mean(distance.cdist(X,Y,'euclidean'))
    return epsilon
#######################  Read the stream  #####################################
# 2 file? one for time one for stream, make it one later
csvName = 'monthbefore.csv'
stream = FileStream(csvName)
df_dat = pd.read_csv(csvName)
df = df_dat.to_numpy()

#Col1,2 has a table
Col1 = 1
Col2 = 0
timeColNum = 2

######################  for all data points in the window  ####################
##gnw is the window counter if 1 --> first window
gnw = 1
X = []
## if we have more sample:
#while(gnw<1500):  
while(stream.has_more_samples()):
    
######################   INITIALIZATION: Window 1 onwards   ###################
    if gnw == 1:        
        c = df[:,2].reshape((-1,1))
        ##first time start_time is the first data point's time
        start_time = c[0]
        time_period = 3600*10 #second

        windowsize = getwindowsize(start_time, time_period)
#        windowsize = 100
        X = stream.next_sample(windowsize)
        
        T = X[0]
        a = T[:,Col1].reshape((-1,1))
        b = T[:,Col2].reshape((-1,1))
        c = df[:,timeColNum].reshape((-1,1))

        X0 = np.concatenate((a,b), axis=1)
        # Compute Affinity Propagation for the first window(X0)
        af = AffinityPropagation(preference=-2.5, damping=.9, max_iter= 100).fit(X0)
        #af = AffinityPropagation(preference=-25, damping=.56, max_iter= 100 ).fit(X)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        my_centers = af.cluster_centers_
        my_centers = my_centers.astype(int)
        n_clusters_ = len(cluster_centers_indices)
        # ADD Time to centroids  and keeps in the separate array      
        t_centers = np.ones(len(my_centers))*start_time
        t_centers = t_centers.astype(int)
        
        branch = np.array(X0)

        #eps = getAdaptiveThreshold()
         # set threshold equal to half or mean the maximum euclidean distance
        #eps = np.max(distance.cdist(my_centers,X0,'euclidean'))/2
        eps = getAdaptiveThreshold(my_centers,X0)
        print(eps)
        gnw = gnw +1
    else:
########################  Plot first Window   #######################################
        
        plt.figure(2)
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels == k
            cluster_center = X0[cluster_centers_indices[k]]
            plt.plot(X0[class_members, 0], X0[class_members, 1],
                     col+ '+', markersize=17)
            plt.plot(cluster_center[0], cluster_center[1], 's', 
                     markerfacecolor=col,markeredgecolor='k', markersize=8)   
##############################################################################        
#        plt.plot(my_centers[:,0], my_centers[:,1], 'gs', markeredgecolor='k',
#                 markersize=5) #markerfacecolor=col, 
#        plt.plot(X[:, 0], X[:, 1],  '+', markersize=8)
# #############################################################################
        #X = current window
        windowsize = getwindowsize(start_time, time_period)
#        windowsize = 18500
        start_time = start_time + time_period
  
        if windowsize == 0:
            X = stream.next_sample(1)
            gnw = gnw +1
            continue
        X = stream.next_sample(windowsize)
        T = X[0]
        a = T[:,Col1].reshape((-1,1))
        b = T[:,Col2].reshape((-1,1))

        X = np.concatenate((a,b), axis=1) 
        
#################      Test Data    ###########################################       
#         df
    #    X = X + np.random.normal(0,0.5,np.shape(X)) #Noise added
    #    X = X + np.sin(gnw)+ np.random.normal(0,0.00025,np.shape(X)) # oscillating cluster location    
    #    X = X*2*np.sin(gnw)                            # Changing cluster size,
    #    X = X*gnw                            # Changing cluster size,
#        sft = gnw*0.91 + np.random.normal(0,0.007,1) #concept-drift
#        X = X + sft  # random Shift added    
###############################################################################
##################      plot Windows  #########################################        
        plt.figure(1)
        plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            class_members = labels == k
            cluster_center = X0[cluster_centers_indices[k]]
            plt.plot(X0[class_members, 0], X0[class_members, 1], 
                     col+ '+', markersize=8)
            plt.plot(cluster_center[0], cluster_center[1], 's',
                     markerfacecolor=col, markeredgecolor='k', markersize=5)   
#            tup_cent = (my_centers,len(class_members))
        plt.plot(my_centers[:,0], my_centers[:,1], 'gs',
                 markeredgecolor='k', markersize=5) #markerfacecolor=col, 
        plt.plot(X[:, 0], X[:, 1],  '+', markersize=8)
        plt.pause(0.02)

##############    usually for the last window with less number of points ######        
        if np.size(X, axis =0) < windowsize: #count the x-->raws
            break
#######################   Comparison STEP   ###################################
# Distance calculation:each data point from centroids #####
        outs = 0
        outpoint = []
        for i in range(windowsize):
            ed = [0]*n_clusters_   

            nxpnt = X[i].reshape(1,2)
            #Distance Arry of each datapoint and all centroids
            ed = distance.cdist(my_centers,nxpnt,'euclidean')     
            
            #min all distances find the closest centroid
#            clclusind = np.argmin(ed) # index of closest cluster
            if min(ed) > eps:
                outs = outs +1
                outpoint.append(X[i])
    #            print("TEST***************")
            else:
                #branch = np.concatenate((branch , np.reshape(X[i,:],(-1,2))))
                ind = np.argmin(ed)          
                t_centers[ind] = start_time
        
################  AP on Outs - ACTIVATEAP   ###################################                
        if outs > 0:
            print(outs)
            Y=np.array(outpoint)        
            f = AffinityPropagation(preference=-0, damping=.99, max_iter= 100 ).fit(Y)
            out_centers = f.cluster_centers_
            #abel_out = f.labels_
    #        my_centers = out_centers
            my_centers = np.append(my_centers,out_centers,axis=0)
            new_t_center = np.ones(len(out_centers))*start_time
            new_t_center = new_t_center.astype(int)
#            out_start_time = np.concatenate((out_centers, new_t_center.T), axis=1)
            t_centers = np.concatenate((t_centers, new_t_center), axis=0)
    #            cluster_center = X[cluster_centers_indices]
            plt.plot(Y[:, 0], Y[:, 1], 'r+', markersize=8)
            plt.plot(out_centers[:,0], out_centers[:,1], 'bs')
            rtb=1
            
            # set interim threshold equal to half the maximum euclidean distance of the reposditry points from the clusters
            int_eps = getAdaptiveThreshold(out_centers,Y)
            
#           int_eps = np.max(distance.cdist(out_centers,Y,'euclidean')) # Get interim threshold values from repository clusters
            print('int_eps',int_eps)
         ## Update cluster threshold size at the end of each window
            if eps < int_eps:
               eps = (int_eps + eps)/2  # If the new threshold is larger than the earlier one then increase the size
    #        else:
    #           eps = (int_eps + eps)/2 # If the new thresholds are smaller then this indicates 
                                   # that there might be smaller clusters away from the original clusters            
        gnw = gnw +1
        print(eps)
    damp = 5
    remain_centers = np.ones(len(t_centers))
    for i in range(len(t_centers)):
        if t_centers[i] < start_time- damp*time_period:
            remain_centers[i] = 0
    my_centers = my_centers[remain_centers==1]
    t_centers = t_centers[remain_centers==1]
    
    windowsize = getwindowsize(start_time, time_period*2)
    n_remaining_samples = stream.n_remaining_samples()
    if(n_remaining_samples <= windowsize):
        break
    
    
        
    
    
### Macro Cluster after AP restart    
#wtmic = # weghted macro
#af_mac = AffinityPropagation(preference= -8, damping=.6, max_iter= 100 ).fit(my_centers) # during one week
af_mac = AffinityPropagation(preference= -4, damping=.99, max_iter= 100 ).fit(my_centers)

cluster_centers_indices_mac = af_mac.cluster_centers_indices_
cluster_centers_mac = af_mac.cluster_centers_


remaining_samples = stream.next_sample(n_remaining_samples)
T = remaining_samples[0]
a = T[:,Col1].reshape((-1,1))
b = T[:,Col2].reshape((-1,1))
remaining_samples = np.concatenate((a,b), axis=1) 
my_label = af_mac.predict(remaining_samples)
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
plt.scatter(df[:,1],df[:,0], s= 2, edgecolor="silver" )
plt.scatter(my_centers[:,0], my_centers[:,1], linewidth=2,facecolors='none', s=100, edgecolor="royalblue")
plt.scatter(cluster_center[:,0],cluster_center[:,1], c='r',marker='+', s=1500, alpha=0.9)
#plt.scatter(1,5, c='r',marker='+', s=1500, alpha=0.9)

#plt.plot(cluster_center[:,0],cluster_center[:,1], '+' , markerfacecolor='r', markersize=80 )


plt.title('DSAP Micro-Macro Clustering Result:Before Intervention')
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
###############################################################################

#3D Evolution 


#from mpl_toolkits.mplot3d import axes3d
#from matplotlib import cm
#
#x = my_centers[:,0]
#y = my_centers[:,1]
#tm = df_dat.Timestamp
#z = np.array(tm)
#omn = np.ones(len(x))
#on = np.array([omn])
#Z = on.T@z
#X,Y = np.meshgrid(x,y)
#print(Z.shape)
#
#fig = plt.figure(figsize=(30,10))
#ax = fig.add_subplot(111, projection='3d')
#
## Plot a 3D surface/scatter
##ax.plot_surface(Z, Y, X, cmap=cm.coolwarm,
#plt.fig(4)                 #    linewidth=0, antialiased=False)
#ax.scatter(z, y, x, c = x, marker='o', s=300, cmap="Spectral")
#p = ax.scatter(z,y,x, c = x, marker='o', s=300, cmap="Spectral")
#fig.colorbar(p)
#ax.set_xlabel('Time(Hr)')
#ax.set_ylabel('Level Number')
#ax.set_ylabel('Cluster Centers-Num of People')
#ax.set_title('Variation of Cluster Centers with Time')
#ax.view_init(88, 270)
#plt.show()
#fig.savefig('ClusterCentCount_3.png',bbox_inches='tight', dpi=400)


    
# #############################################################################
# =============================================================================
print("Processing time: %s seconds" % (time.time() - start_time_Org))

#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')

cProfile.run('re.compile("foo|bar")')

# =============================================================================


    
#print('Silhouette Coefficient Micro:',metrics.silhouette_score(Y, abel_out, metric='euclidean'))
#print('Davies-Bouldin Index Micro:',davies_bouldin_score(Y, abel_out))
#print('Calinski-Harabasz Index Micro:',metrics.calinski_harabasz_score(Y, abel_out))
###############################################################################
print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(remaining_samples, my_label))  
print('Silhouette Coefficient Macro:',metrics.silhouette_score(remaining_samples, my_label, metric='euclidean'))
print('Davies-Bouldin Index Macro:',davies_bouldin_score(remaining_samples, my_label))

###############################################################################
#cProfile.run('re.compile("foo|bar")')

