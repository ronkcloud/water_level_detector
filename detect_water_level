import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from collections import defaultdict
import random as rd
import csv

class WaterLevelDetector(object):

    filter = {
        'derivfl3': np.array([
                        [1],
                        [0],
                        [-1]]),

        'derivfl5': np.array([
                        [1],
                        [1],
                        [0],
                        [-1],
                        [-1]]),

        'boxfl5': np.array([
                        [1],
                        [1],
                        [1],
                        [1],
                        [1]]),

        'gausfl5': np.array([
                        [0],
                        [1],
                        [3],
                        [1],
                        [0]]),

        'gausfl7': np.array([
                        [0],
                        [1],
                        [3],
                        [5],
                        [3],
                        [1],
                        [0]])

    }

    def __init__(self, path, pts1, currentWaterLevel=0, bevel=100, peilschaalLen=1000, obsRead=0, folderPath=False, display=False):
        currentPeilschaalLen = peilschaalLen - currentWaterLevel

        image = self.readImage(path)
        undistort = self.undistortImage(image)
        warpedImage = self.WarpImage(undistort, currentPeilschaalLen, bevel, peilschaalLen)
        avgRows = self.avgRows(warpedImage)
        variance = self.variance(avgRows, self.filter['gausfl7'])
        self.clusters, centroids = self.kMeans(avgRows, variance, 2)
        segments = self.segmentByKMeans(self.clusters, avgRows)
        self.kMeanspointer, self.status = self.detectWaterLevelbyKMeans(segments, self.filter['derivfl3'], peilschaalLen)
        thresholdImage = self.adaptiveThreshold(warpedImage)
        avgThresholdRows = self.avgRows(thresholdImage)
        self.thresholdPointer, gausFiltered, derivFiltered = self.detectWaterLevelbyThreshold(avgThresholdRows, peilschaalLen)
        
        if (display == True):
            self.display(warpedImage, thresholdImage, avgRows, variance, centroids, self.clusters, segments, avgThresholdRows, gausFiltered, derivFiltered)

    def display(self, warpImg, thImg, avgRows, variance, Centroids, Output, segments, avgThRows, gausFiltered, derivFiltered):

        color=['red','blue','green','cyan','magenta']
        labels=['cluster1','cluster2','cluster3','cluster4','cluster5']

        h = len(avgRows) 
        y_axis = list(range(h))

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2,8)
        fg_ax1 = fig.add_subplot(gs[0,0])
        fg_ax1.imshow(warpImg)
        fg_ax1.set_title('warp_img')
        fg_ax2 = fig.add_subplot(gs[0,4])
        fg_ax2.imshow(thImg)
        fg_ax2.set_title('th_img')
        fg_ax3 = fig.add_subplot(gs[0,1])
        fg_ax3.plot(avgRows, y_axis)
        fg_ax3.set_title('avg_rows')
        fg_ax3.set(ylim=(h,0))
        fg_ax4 = fig.add_subplot(gs[0,3])
        fg_ax4.plot(segments, y_axis)
        fg_ax4.set_title('clustered')
        fg_ax4.set(ylim=(h,0))
        fg_ax6 = fig.add_subplot(gs[0,2])
        fg_ax6.plot(variance, y_axis)
        fg_ax6.set_title('variance')
        fg_ax6.set(ylim=(h,0))
        fg_ax10 = fig.add_subplot(gs[1,0:3]) 
        for k in range(2):
            fg_ax10.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
        fg_ax10.scatter(Centroids[0,:],Centroids[1,:],s=300,c='yellow',label='Centroids')
        fg_ax10.set_title('img_histogram vs variance (clustered)')
        fg_ax10.set_xlabel('img_histogram')
        fg_ax10.set_ylabel('variance')
        fg_ax9 = fig.add_subplot(gs[0,5])    
        fg_ax9.plot(avgThRows, y_axis)
        fg_ax9.set_title('avg_th_rows')
        fg_ax9.set(ylim=(h,0))
        fg_ax11 = fig.add_subplot(gs[0,6])
        fg_ax11.plot(gausFiltered, y_axis)
        fg_ax11.set_title('gaussian_filter')
        fg_ax11.set(ylim=(h,0))
        fg_ax12 = fig.add_subplot(gs[0,7])
        fg_ax12.plot(derivFiltered, y_axis)
        fg_ax12.set_title('derivation_filter')
        fg_ax12.set(ylim=(h,0))
        plt.show()

    def getWaterLevel(self):
        result = self.kMeanspointer
        if (self.status == False):
            result = self.thresholdPointer
        return result

    def adaptiveThreshold(self, image):
        greyscaleImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.adaptiveThreshold(greyscaleImage,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,23,4)

    def detectWaterLevelbyThreshold(self, data, peilschaalLen):
        filter1 = self.conv1d(data, self.filter['gausfl5'])
        filter2 = self.deriv1d(filter1, self.filter['derivfl5'])
        return peilschaalLen-np.argmax(filter2), filter1, filter2

    def segmentByKMeans(self, clusters, data):
        segments = np.copy(data)
        for segment in segments:
            for cluster in clusters[1]:
                if segment[0] == cluster[0]:
                    segment[0] = 0
            if segment[0]!= 0:
                segment[0] = 255
        return segments
    
    def detectWaterLevelbyKMeans(self, segments, krnl, peilschaalLen):
        segmentPoint = self.deriv1d(segments, krnl)
        segmentPoint[0:int(peilschaalLen/10)] = 0
        segmentPoint[len(segments)-int(peilschaalLen/10):len(segments)] = 0
        points, indices, counts = np.unique(segmentPoint, return_counts=True, return_index=True)
        pointCount = counts[np.argmax(points)]
        pointer = indices[np.argmax(points)]
        segmentationStatus = True
        if (pointCount > 4):
            segmentationStatus = False
        return peilschaalLen-pointer, segmentationStatus

    def kMeans(self, hist, val, K):
        X = np.c_[hist, val]

        m=X.shape[0]
        Centroids=np.array([]).reshape(2,0)

        #step 1
        for i in range(K):
            rand=rd.randint(0,m-1)
            Centroids=np.c_[Centroids,X[rand]]

        #step2
        num_iter=100
        Output=defaultdict()
        Output={}
        itercount = 0
        for n in range(num_iter):
            #step 2.a
            EuclidianDistance=np.array([]).reshape(m,0)
            for k in range(K):
                tempDist=np.sum((X-Centroids[:,k])**2,axis=1)
                EuclidianDistance=np.c_[EuclidianDistance,tempDist]
                
            C=np.argmin(EuclidianDistance,axis=1)+1
            #step 2.b
            Y={}
            for k in range(K):
                Y[k+1]=np.array([]).reshape(2,0)
            for i in range(m):
                Y[C[i]]=np.c_[Y[C[i]],X[i]]
            
            for k in range(K):
                Y[k+1]=Y[k+1].T
                
            for k in range(K):
                Centroids[:,k]=np.mean(Y[k+1],axis=0)   

        return Y, Centroids

    def avgRows(self, image):
        avgRows = []
        for rows in image: 
            avgRows.append(np.mean(rows))
        return np.reshape(avgRows, (len(avgRows),1))

    def addPadding(self, data, xpad, ypad):
        a = data[0][0]
        b = [[data[len(data)-1][0]]]
        for i in range(ypad):
            data = np.insert(data, 0, a,axis=0)
        for i in range(ypad):
            data = np.append(data, b, axis=0)
        return data,xpad,ypad

    def deriv1d(self, data, krnl):
        ypad = int((len(krnl)/2)-0.5)
        pad, xpad, ypad = self.addPadding(data, 0, ypad)
        acc = np.zeros((len(data),1))
        for y in range(ypad,len(pad)-ypad):
            for x in range (len(pad[0])):
                k = pad[y-ypad:y+ypad+1]
                r = np.multiply(krnl,k)
                acc[y-ypad][x] = (np.sum(r))**2
        return acc

    def variance(self, data, krnl):
        ypad = int((len(krnl)/2)-0.5)
        pad, xpad, ypad = self.addPadding(data,0,ypad)
        acc = np.zeros((len(data),1))
        for y in range(ypad,len(pad)-ypad):
            for x in range (len(pad[0])):
                k = pad[y-ypad:y+ypad+1]
                kmin = min(k)
                kmax = max(k)
                acc[y-ypad][x] = abs(kmin-kmax)
        return acc

    def conv1d(self, data, krnl):
        ypad = int((len(krnl)/2)-0.5)
        pad, xpad, ypad = self.addPadding(data,0,ypad)
        acc = np.zeros((len(data),1))
        for y in range(ypad,len(pad)-ypad):
            for x in range (len(pad[0])):
                k = pad[y-ypad:y+ypad+1]
                r = np.multiply(krnl,k)
                acc[y-ypad][x] = np.mean(r)
        return acc
    
    def WarpImage(self, dst, currentPeilschaalLen, bevel, peilschaalLen):
        peilschaalLen_B = peilschaalLen+bevel
        peilschaalWidth = int((pts1[1][0] - pts1[0][0])*currentPeilschaalLen/(pts1[2][1]-pts1[1][1]))
        pts2 = np.float32([[0,0],[peilschaalWidth,0],[0,currentPeilschaalLen],[peilschaalWidth,currentPeilschaalLen]])
        pts3 = np.float32([[0,0],[peilschaalWidth,0],[0,peilschaalLen_B],[peilschaalWidth,peilschaalLen_B]])
        M1 = cv2.getPerspectiveTransform(pts1,pts2)
        ret, IM = cv2.invert(M1)
        pts3_2 = pts3.reshape(-1,1,2)
        pts3_2 = cv2.perspectiveTransform(pts3_2, IM)
        M = cv2.getPerspectiveTransform(pts3_2,pts3)
        dst2 = cv2.warpPerspective(dst,M,(peilschaalWidth, peilschaalLen_B))
        return dst2

    def readImage (self, path):
        return cv2.imread(path)

    def undistortImage(self, image):
        mtx = np.loadtxt('mtx.csv')
        dist = np.loadtxt('dist.csv')
        h, w = image.shape[:2]
        newCameraMtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(image, mtx, dist, None, newCameraMtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        return dst
    
pts1 = np.float32([[1065,497],[1133,496],[1077,940],[1133,940]])
path = ./asset/image1.jpg
detect = WaterLevelDetector(path, pts1, currentWaterLevel=0, bevel=20, peilschaalLen=100, display = False)
waterLevel = detect.getWaterLevel()
