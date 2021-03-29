# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:22:01 2020

@author: neshragh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AffinityPropagation
from PIL import Image, ImageDraw, ImageFont




df = pd.read_csv('TrainingData_order_withoutSignal.csv')

# =============================================================================
# isn =df.isnull().sum()
# print(isn)
# print(df.isnull())
# =================

#Building Map

#df_1 = df[(df['FLOOR'] == 1) & (df['BUILDINGID'] == 0)]
#plt.scatter(df_1.LONGITUDE,df_1.LATITUDE)
#df_1 = df[(df['FLOOR'] == 1) & (df['BUILDINGID'] == 1)]
#plt.scatter(df_1.LONGITUDE,df_1.LATITUDE)
#df_1 = df[(df['FLOOR'] == 1) & (df['BUILDINGID'] == 2)]
#plt.scatter(df_1.LONGITUDE,df_1.LATITUDE)
#
##generate building map in the loop
#for i in range(0,4):
#    for j in range(0,2):
#        df_1 = df[(df['FLOOR'] == i) & (df['BUILDINGID'] == j)]
#        plt.scatter(df_1.LONGITUDE,df_1.LATITUDE, color= 'firebrick')
#        #show seprately
#        plt.show()
        

#Drop Empty Rows or Columns
#df.dropna(axis=0,how='any', thresh=None, subset=None, inplace=True)

#drop with criteria
#df = df[df.WAP001 == '100']

    
#for i in range(0,3):
#    b = df.iloc[:, i]
#    print(b)
#    con = b.iloc[i] == 100
#    if con:
#        con.drop()

#plt.scatter(df.BUILDINGID, df.FLOOR)



#############################################################################
#c = df.LATITUDE
#b = df.FLOOR
#a = df.LONGITUDE
#
#
#
############# 3D Plotting 
#fig = plt.figure(2)
##ax = Axes3D(fig)
##for i in range(0,10):
##    ax.scatter(df.LONGITUDE, df.LATITUDE, df.iloc[:,i], s=20)
##plt.xlabel('Long',labelpad=15)
##plt.ylabel('Lat',labelpad=15)
##
##
##plt.show()
##df.astype(int)
#
#
#am = Axes3D(fig)
#am.scatter(c,b, a ,s=20, color= 'royalblue')
#
#
#am.set_xlabel('Latitude',labelpad=10)
#am.set_ylabel('Floor',labelpad=10)
#am.set_zlabel('Longitude',labelpad=10)
#
#plt.show()

###############################################################################
"""
BUILDINGID
LATITUDE
LONGITUDE
PHONEID
SPACEID
FLOOR

RELATIVEPOSITION
"""

c = df.LATITUDE
b = df.LONGITUDE
a = df.PHONEID

############ 3D Plotting 
fig = plt.figure(2)
am = Axes3D(fig)
pl = am.scatter(a,b,c ,s=20,  c = a, cmap = 'jet')

#am.set_xlabel('Latitude',labelpad=10)
#am.set_ylabel('Floor',labelpad=10)
#am.set_zlabel('Longitude',labelpad=10)

fig.colorbar(pl, shrink = 0.75)
am.view_init(50, 0)
plt.show()




























