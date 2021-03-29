
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
import os
import cProfile
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

csvName = '1weekafter.csv'
stream = FileStream(csvName)
df_dat = pd.read_csv(csvName)
df = df_dat.to_numpy()

#Col1,2 has a table
Col1 = 1
Col2 = 0
timeColNum = 2
###############################################################################
######################  for all data points in the window  ####################

gnw = 0    
c = df[:,2].reshape((-1,1))
start_time = c[0]
X = []
#while(gnw<1500):  
while(stream.has_more_samples()):
    time_period = 3600*5 #second

    windowsize = getwindowsize(start_time, time_period)
    start_time = start_time + time_period
#    windowsize = 100
    if windowsize <3:
        X = stream.next_sample(1)
        continue
    else:
        gnw = gnw +1
        
######################   INITIALIZATION: Window 1 onwards   ###################
    if gnw == 1:        
        X = stream.next_sample(windowsize)
        
        T = X[0]
        a = T[:,Col1].reshape((-1,1))
        b = T[:,Col2].reshape((-1,1))
        c = df[:,timeColNum].reshape((-1,1))

        X0 = np.concatenate((a,b), axis=1)
        # Compute Affinity Propagation for the first window(X0)
        af = AffinityPropagation(preference=-2.5,
                                 damping=.9, max_iter= 100).fit(X0)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        my_centers = af.cluster_centers_
        my_centers = my_centers.astype(int)
        n_clusters_ = len(cluster_centers_indices)
        # ADD Time to centroids  and keeps in the separate array 
        #size is equal to my_centroids
        # time for all is equal to start time caz all of them are in one window
        t_centers = np.ones(len(my_centers))*start_time
        t_centers = t_centers.astype(int)
        
        branch = np.array(X0)
        branch_label= np.array(labels)
         # set the initial threshold equal to mean the maximum euclidean distance
        eps = getAdaptiveThreshold(my_centers,X0)
#        print(eps)
        gnw = gnw +1
########################  Plot 1st Window   #######################################
        
#        plt.figure(1)
#        plt.clf()
#        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#        for k, col in zip(range(n_clusters_), colors):
#            class_members = labels == k
#            cluster_center = X0[cluster_centers_indices[k]]
#            plt.plot(X0[class_members, 0], X0[class_members, 1],
#                     col+ '+', markersize=17)
#            plt.plot(cluster_center[0], cluster_center[1], 's', 
#                     markerfacecolor=col,markeredgecolor='k', markersize=8)   
###############################################################################        
#        plt.plot(my_centers[:,0], my_centers[:,1], 'gs', markeredgecolor='k',
#                 markersize=5) #markerfacecolor=col, 
#        plt.plot(X[:, 0], X[:, 1],  '+', markersize=8)
# ###########   From second window and so on ...  #############################
    else:
        # Now we have data points in the window
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
#        plt.figure(2)
#        plt.clf()
#        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#        for k, col in zip(range(n_clusters_), colors):
#            class_members = labels == k
#            cluster_center = X0[cluster_centers_indices[k]]
#            plt.plot(X0[class_members, 0], X0[class_members, 1], 
#                     col+ '+', markersize=14)
#            plt.plot(cluster_center[0], cluster_center[1], 's',
#                     markerfacecolor=col, markeredgecolor='k', markersize=10)   
##            tup_cent = (my_centers,len(class_members))
#        plt.plot(my_centers[:,0], my_centers[:,1], 'gs',
#                 markeredgecolor='k', markersize=5) #markerfacecolor=col, 
#        plt.plot(X[:, 0], X[:, 1], '+', markersize=8)
#        plt.pause(0.04)

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
            idx = np.argmin(ed)

            #min all distances find the closest centroid
#            clclusind = np.argmin(ed) # index of closest cluster
            if min(ed) > eps:
                outs = outs +1
                outpoint.append(X[i])
    #            print("TEST***************")
            else:
                branch = np.concatenate((branch , np.reshape(X[i,:],(-1,2))))
                branch_label= np.concatenate((branch_label , np.reshape(idx,(-1))))

                ind = np.argmin(ed)          
                t_centers[ind] = start_time
        
################  AP on Outs - ACTIVATEAP   ###################################                
        if outs > 0:
#            print(outs)
            Y=np.array(outpoint)        
            f = AffinityPropagation(preference=-0,
                                    damping=.99, max_iter= 100 ).fit(Y)
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
#            print('int_eps',int_eps)
         ## Update cluster threshold size at the end of each window
         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if eps < int_eps:
               eps = (int_eps + eps)/2  # If the new threshold is larger than the earlier one then increase the size
         #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #        else:
    #           eps = (int_eps + eps)/2 # If the new thresholds are smaller then this indicates 
                                   # that there might be smaller clusters away from the original clusters            
        gnw = gnw +1
#        print(eps)
        
################   Calculate the fading  ######################################
      
    faid = 50
    remain_centers = np.ones(len(t_centers))
    for i in range(len(t_centers)):
        if t_centers[i] < start_time- faid * time_period:
            remain_centers[i] = 0
    my_centers = my_centers[remain_centers==1]
    t_centers = t_centers[remain_centers==1]
    
###########   Calculating the indexes based on the n latest windows ###########    
#    windowsize = getwindowsize(start_time, time_period*100)
#    n_remaining_samples = stream.n_remaining_samples()
#    if(n_remaining_samples <= windowsize):
#        break
    
    
        
    
###############################################################################
################# Offline Macro-Clustering  ###################################
###############################################################################
af_mac = AffinityPropagation(preference= -4,
                             damping=.9, max_iter= 100 ).fit(my_centers)
cluster_centers_indices_mac = af_mac.cluster_centers_indices_
cluster_centers_mac = af_mac.cluster_centers_

#This is for calculating the indexes based on the remaining sample
#remaining_samples = stream.next_sample(n_remaining_samples)
#T = remaining_samples[0]
#a = T[:,Col1].reshape((-1,1))
#b = T[:,Col2].reshape((-1,1))
#remaining_samples = np.concatenate((a,b), axis=1) 
#my_label = af_mac.predict(remaining_samples)
my_label = af_mac.labels_

###############################################################################
########################   Plot   #############################################

plt.figure(3)
plt.clf()
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
cluster_center = my_centers[cluster_centers_indices_mac]

#plt.scatter(df[:,0],df[:,1])
#plt.scatter(df[:,1],df[:,0], s= 2, edgecolor="silver" )
plt.scatter(my_centers[:,0], my_centers[:,1], linewidth=2,
            facecolors='none', s=100, edgecolor="royalblue")
plt.scatter(cluster_center[:,0],cluster_center[:,1], c='r',
            marker='+', s=1500, alpha=0.9)
#plt.scatter(1,5, c='r',marker='+', s=1500, alpha=0.9)
#plt.plot(cluster_center[:,0],cluster_center[:,1], '+' , markerfacecolor='r', markersize=80 )
#plt.title('DSAP Micro-Macro Clustering: March 18- March 24')
#plt.title('DSAP Micro-Macro Clustering: April 29- May 5')
plt.title('DSAP Micro-Macro Clustering: May 27- June 2')

#plt.title('%d' % n_clusters_)
plt.xlabel('Position')
plt.ylabel('Count')
frame =plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)
#plt.ylim(0,20)
plt.xticks([1,2,3,4,5,6], ["Level 2-1", "Level 3-2\nCentral", "Level 4-3\nNorth",
           "Level4-3\nSouth", "Level5-4\nNorth", "Level5-4\nSouth"])
plt.show()
###############################################################################
# Number of micro and macro Clusters 
n_cluster_ = len(cluster_centers_indices_mac)
print('Estimated number of macro clusters: %d' % n_cluster_)
n_cluste = len(my_centers)
print('Estimated number of micro clusters: %d' % n_cluste)
###############################################################################
#############################   Evaluation Phase ##############################
###############################################################################
print("Processing time: %s seconds" % (time.time() - start_time_Org))
timeprocess= (time.time() - start_time_Org)
#Profiling and memory usage--------------------------------------------------
process = psutil.Process(os.getpid())
print("Memory Consumption:",process.memory_info().rss, 'bytes')   
M= process.memory_info().rss /1000000
print('(Or Megabyte:', M,')')

#cProfile.run('re.compile("foo|bar")')

###############################################################################
#print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(remaining_samples, my_label))  
#print('Silhouette Coefficient Macro:',metrics.silhouette_score(remaining_samples, my_label, metric='euclidean'))
#print('Davies-Bouldin Index Macro:',davies_bouldin_score(remaining_samples, my_label))

###############################################################################
print('Calinski-Harabasz Index Macro:',metrics.calinski_harabasz_score(my_centers, my_label))  
print('Silhouette Coefficient Macro:',metrics.silhouette_score(
        my_centers, my_label, metric='euclidean'))
print('Davies-Bouldin Index Macro:',davies_bouldin_score(my_centers, my_label))

###############################################################################
###############  Save metrics in excel file ###################################

timeprocess=np.array(timeprocess)
M=np.array(M)
d=[timeprocess,M]
db =davies_bouldin_score(my_centers, my_label)
sil= metrics.silhouette_score(my_centers, my_label, metric='euclidean')
cal =metrics.calinski_harabasz_score(my_centers, my_label)
mic_c = n_cluste
mac_c = n_cluster_


dmac= pd.DataFrame(data=([[windowsize, timeprocess, M, sil, cal, db, mic_c,
                           mac_c]]), columns=['windowSize', 'timeprocess','M', 
                    'silhouette', 'calinski', 'davies_bo', 'micro', 'macro'])

dmac.to_csv(r'evaluate.csv', index = False, 
            header=(not os.path.exists('evaluate.csv')), mode='a')






