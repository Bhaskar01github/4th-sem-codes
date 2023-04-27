
def cen(a):
    b=len(a)
    su=0
    c=[]
    for x in range(0,2):
     su=0
     for i in range(0,b):
        su=su+a[i][x]
     c.append(su/b)  
    return c
def distance(a,b):
    import math
    c=math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    return c
import matplotlib.pyplot as plt
import sklearn

x = [26.18, 22.96, 25.38, 22.73,21.00, 22.74, 9.38 , 12.12, 11.79]
y = [91.75, 88.57, 86.47, 71.64,73.30, 77.74, 76.57, 76.68, 77.80]

plt.scatter(x, y)
plt.show()
from sklearn.cluster import KMeans

data = list(zip(x, y))
print(data)
location={data[0]:"guwahati(assam)",data[1]:"haringhata(wb)",data[2]:"harnaut(bihar)",data[3]:"surendranath(gujarat)",data[4]:"ratnagiri(maharastra)",data[5]:"Hoshananad (MP) ",data[6]:"Thiruvalla (Kerala)",data[7]:"Mysore (Karnataka)",data[8]:"Mettur (TN)" }
inertias = []
for i in range(1,9):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,9), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

plt.scatter(x, y, c=kmeans.labels_)
plt.show()
labels=kmeans.predict(data)
print(labels)
dictionary={}
for i in range(len(labels)):
    if labels[i] in dictionary:
        dictionary[labels[i]].append(data[i])
    else:
        dictionary[labels[i]]=[data[i]]
print(dictionary) 
centroid={}
for i in range(3):
   centroid[i]=cen(dictionary[i])
print(centroid)       
new=input("enter the location (latitude,longitude): ").split(",")
a=[float(new[0]),float(new[1])]
dis={}
for i in range(3):
    dis[i]=distance(centroid[i],a)
print("this is dis ",dis)    
find=0
for i in range(len(dis)):
    if dis[find]>dis[i]:
        find=i
    else:
        find=find  
for i in range(len(dictionary[find])):
    print(location[dictionary[find][i]])

#with module
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 17:39:19 2023

@author: sk21ms051
"""
'''
import matplotlib.pyplot as plt
x = [4, 5, 10, 4, 3, 11, 14 , 8, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
classes = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
plt.scatter(x,y,c=classes)
plt.show
from sklearn.neighbors import KNeighborsClassifier
data=list(zip(x,y))
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data,classes)
new_x = 8
new_y = 21
new_point = [(new_x, new_y)]
prediction = knn.predict(new_point)
print(prediction)
plt.scatter(x + [new_x], y + [new_y], c=classes + [prediction[0]])
plt.text(x=new_x-1.7, y=new_y-0.7, s=f"new point, class: {prediction[0]}")
plt.show()
'''

    