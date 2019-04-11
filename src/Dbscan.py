import numpy as np

'''
This class is used to run the DBSCAN algorithm
This is based on the following paper: 
https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
'''
class Dbscan():
    def __init__(self, eps, minSamples):
        self.__eps = eps
        self.__minSamples = minSamples
        self.labels = []
        self.__cId = 0
        
    
    '''
    This is the main function that is used in this class.
    This clusters the points provided based on the class globals such as 
    eps and minSamples
    '''
    def fit(self, points):
        # Labels that track class of each point and a moving class id
        # that increments for new clusters (and is -1 for noise)
        self.labels = [None] * len(points)
        
        # Visit all points
        for pt in range(len(points)):
            
            if(self.labels[pt] != None):
                # Already classified
                continue
            
            nbrs = self.__getNeighbourPoints(points, pt)
            
            # Check if we found a core, border or noise point
            if(len(nbrs) < self.__minSamples):
                # Noise point, or possibly a border point
                self.labels[pt] = -1
            elif(len(nbrs) >= self.__minSamples):
                # p is a core point and the cluster is valid
                # we need to know the extent of the cluster
                self.__expandCluster(points, pt, nbrs)
                self.__cId += 1
                
        return self.labels

    '''
    This function is used to get the full extent of a cluster
    and is called on a point we know to be a core point
    '''
    def __expandCluster(self, points, pCoreIndex, nbrs): 
        # Core is part of cluster
        self.labels[pCoreIndex] = self.__cId
        
        
        while(nbrs != []):
            # We visit whatever is left
            visitPointIndex = nbrs[0]            
            
            # If it's noise, it is now a border point
            if(self.labels[visitPointIndex] == -1):
                self.labels[visitPointIndex] = self.__cId
                
            # If we haven't checked it before, we can check if it's core
            elif(self.labels[visitPointIndex] == None):
                # If or if it is not core, it is still part of cluster so we add it
                self.labels[visitPointIndex] = self.__cId
                
                visitPointNbrs = self.__getNeighbourPoints(points, visitPointIndex)
                
                # If it has enough neighbours, it is core 
                if(len(visitPointNbrs) >= self.__minSamples):
                    # So we visit all its neigbhours
                    nbrs += visitPointNbrs
                
            # We're done with this point now so take it out of loop
            nbrs = nbrs[1:]
        

    '''
    Calculates the distance between two points based on euclidean function
    '''
    def __calculateDistance(self, p1, p2):
        return np.linalg.norm(p1-p2)

    '''
    This function gets all the (indexes of) neighbours of a point, pCore 
    A point is a neighbour of pCore if it is
    1. In allP (all points to consider) and
    2. Has a euclidean distance less than eps
    '''
    def __getNeighbourPoints(self, allP, pCoreIndex):
        c = []
        for n in range(len(allP)):
            dist = self.__calculateDistance(allP[pCoreIndex], allP[n])
            
            # If point is a neighbour (based on eps) we add index to cluster
            if(dist <= self.__eps):
                c.append(n)
        return c