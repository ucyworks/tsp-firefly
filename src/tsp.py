import numpy as np
import math
import os

class TSPProblem:
    """
    Class representing a Traveling Salesman Problem instance.
    """
    def __init__(self, instance_name=None, file_path=None):
        """
        Initialize a TSP problem instance.
        
        Parameters:
        -----------
        instance_name : str
            Name of the problem instance
        file_path : str
            Path to the TSP instance file
        """
        self.name = instance_name
        self.dimension = 0
        self.coordinates = []
        self.distance_matrix = None
        
        if file_path and os.path.exists(file_path):
            self.load_problem(file_path)
    
    def load_problem(self, file_path):
        """
        Load a TSP problem from a file in TSPLIB format.
        
        Parameters:
        -----------
        file_path : str
            Path to the TSP instance file
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Parse header information
        reading_coords = False
        coords = []
        
        for line in lines:
            line = line.strip()
            if line == "EOF":
                break
                
            if reading_coords:
                # Parse coordinate line
                parts = line.split()
                if len(parts) >= 3:  # id, x, y
                    coords.append((float(parts[1]), float(parts[2])))
            elif line.startswith("NAME"):
                self.name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                self.dimension = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
        
        self.coordinates = coords
        
        # Compute the distance matrix
        self._compute_distance_matrix()
    
    def _compute_distance_matrix(self):
        """Compute the distance matrix for the TSP instance."""
        n = len(self.coordinates)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.coordinates[i]
                    x2, y2 = self.coordinates[j]
                    # Euclidean distance
                    self.distance_matrix[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_distance(self, city1, city2):
        """Get the distance between two cities."""
        return self.distance_matrix[city1][city2]
    
    def get_route_distance(self, route):
        """
        Calculate the total distance of a route.
        
        Parameters:
        -----------
        route : list
            List of city indices representing a route
            
        Returns:
        --------
        float
            Total distance of the route
        """
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.get_distance(route[i], route[i + 1])
        
        # Add distance from last city back to first
        total_distance += self.get_distance(route[-1], route[0])
        
        return total_distance
    
    def generate_random_instance(self, dimension, min_coord=0, max_coord=100):
        """
        Generate a random TSP instance.
        
        Parameters:
        -----------
        dimension : int
            Number of cities
        min_coord : float
            Minimum coordinate value
        max_coord : float
            Maximum coordinate value
        """
        self.name = f"Random{dimension}"
        self.dimension = dimension
        self.coordinates = []
        
        for _ in range(dimension):
            x = min_coord + (max_coord - min_coord) * np.random.random()
            y = min_coord + (max_coord - min_coord) * np.random.random()
            self.coordinates.append((x, y))
        
        self._compute_distance_matrix()
