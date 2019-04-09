
'''
This class is used to run the DBSCAN algorithm
https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf
'''
class Dbscan():
    def __init__(self, eps, minN):
        self.__eps = eps
        self.__minN = minN
        self.__clusters = [[]]
        
    def __calculateDistance(self, p1, p2):
        # Calculates the euclidean distance between 2 points
        return 0 
    
    def fit(self, points):
        # Takes in a set of points and begins clustering process
        
        return self.__clusters