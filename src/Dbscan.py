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
        self.__clusters = [[]]
        
    def calculateDistance(self, p1, p2):
        # Calculates the euclidean distance between 2 points
        return np.linalg.norm(p1-p2)
    
    def fit(self, points):
        # Takes in a set of points and begins clustering process
        
        unvisited = copy(points)
        
        # While we have points still to visit
        while(unvisited != []):
            # Pick a random point and visit it
            rIndex = randint(0, len(unvisited) - 1)
            p = unvisited[rIndex]
            unvisited.pop(rIndex)
            
            c = self.getNeighbourPoints(unvisited, p)    
            
            # After forming the inital cluster, we check if it is valid 
            if(len(c) < self.__minSamples):
                # Not valid, p is a border point, move on
                continue
            elif(len(c) >= self.__minSamples):
                # p is a core point and the cluster is valid
                # we need ot know the extent of the cluster DFS
                
                # pop all points we have in current neighbourhood from unviisted
                for i in range(1, len(c)):
                    ind = np.where(unvisited == c[i])[0][0]
                    print(ind)
                    #ind = unvisited.index(c[i])
                    unvisited.pop(ind)
                # use that to get more points
                # repeart 1 and 2 with those new points 
                print('s')
            
    def getNeighbourPoints(self, allP, pCore):
        # Determine neighbours of pCore based on eps and minSamples
        c = [pCore]
        for n in range(len(allP)):
            # Cluster starts with the core point
            dist = self.calculateDistance(pCore, allP[n])
            
            # If point is a neighbour (based on eps) we add it to cluster
            if(dist <= self.__eps):
                c.append(allP[n])
        return c
        