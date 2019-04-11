import numpy as np
from copy import copy
from random import randint

'''
This class is used to run the DBSCAN algorithm
https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
'''
class Dbscan():
    def __init__(self, eps, minSamples):
        self.__eps = eps
        self.__minSamples = minSamples
        self.__clusters = []
        
    def calculateDistance(self, p1, p2):
        # Calculates the euclidean distance between 2 points
        return np.linalg.norm(p1-p2)
    
    def fit(self, points):
        # Takes in a set of points and begins clustering process
        
        unvisited = copy(points)
        noise = []
        
        # While we have points still to visit
        while(unvisited != []):
            # Pick a random point and visit it
            rIndex = randint(0, len(unvisited) - 1)
            p = unvisited[rIndex]
            unvisited.pop(rIndex)
            
            cNbrsLen = len(self.getNeighbourPoints(unvisited, p)) + 1
            
            # Check if we found a core, border or noise point
            if(cNbrsLen == 0):
                noise.append(p)
            if(cNbrsLen < self.__minSamples):
                # Not valid, p is a border point, move on
                continue
            elif(cNbrsLen >= self.__minSamples):
                # p is a core point and the cluster is valid
                # we need ot know the extent of the cluster DFS
                newC = self.dfsCluster(copy(unvisited), p)
                
                #print(newC)
#               
                self.__clusters.append(newC)
        return self.__clusters
            
    def getNeighbourPoints(self, allP, pCore):
        # Determine neighbours of pCore based on eps and minSamples
        c = []
        for n in range(len(allP)):
            # Cluster starts with the core point
            dist = self.calculateDistance(pCore, allP[n])
            
            # If point is a neighbour (based on eps) we add it to cluster
            if(dist <= self.__eps):
                c.append(allP[n])
        return c
        
    
    
    def dfsCluster(self, unvisited, pCore):
        # Core should not be part of unvisited 
        # Gets all the neighbours of a core point 
        nbrs = self.getNeighbourPoints(unvisited, pCore)
        
        cluster = copy(nbrs)
        
        for i in range(len(nbrs)):
            # Pop the neighbours from unvisited
            ind = np.where(unvisited == nbrs[i])[0][0]
            unvisited.pop(ind)
        
        
        for i in range(len(nbrs)):
            # Get cluster of neigbhour point and add it to cluster
            cluster += self.dfsCluster(unvisited, nbrs[i])
            
        return cluster
        