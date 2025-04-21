import numpy as np
from tqdm import tqdm
import random
import copy
import multiprocessing
from functools import partial

class FireflyAlgorithm:
    """
    Implementation of Firefly Algorithm for solving TSP.
    """
    def __init__(self, problem, num_fireflies=50, alpha=0.2, beta_0=1.0, gamma=0.01, max_iterations=100, use_parallel=False, num_cores=None):
        """
        Initialize the Firefly Algorithm.
        
        Parameters:
        -----------
        problem : TSPProblem
            The TSP problem instance to solve
        num_fireflies : int
            Number of fireflies in the population
        alpha : float
            Randomization parameter
        beta_0 : float
            Attractiveness at distance = 0
        gamma : float
            Light absorption coefficient
        max_iterations : int
            Maximum number of iterations
        use_parallel : bool
            Whether to use parallel processing
        num_cores : int
            Number of CPU cores to use for parallel processing
        """
        self.problem = problem
        self.num_fireflies = num_fireflies
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.use_parallel = use_parallel
        self.num_cores = num_cores if num_cores else multiprocessing.cpu_count()
        
        # Pre-calculate distance matrix for faster access
        self.distance_matrix = problem.distance_matrix
        
        # Initialize population of fireflies
        self.fireflies = []
        self.intensities = []
        self.best_solution = None
        self.best_distance = float('inf')
        self.convergence = []
        
        # For faster computation
        self.dim = problem.dimension
        self.visited_edges = {}  # Cache for route distances
        
        # Sampling strategy for faster evolution
        self.max_comparisons = min(num_fireflies // 2, 10)
        
    def initialize_population(self):
        """Initialize a population of fireflies with smart initialization."""
        self.fireflies = []
        
        # Include a 2-opt nearest neighbor solution for better starting point
        nn_solution = self._get_nn_solution()
        self.fireflies.append(nn_solution)
        
        # Fill the rest with perturbed nearest neighbor solutions
        for _ in range(self.num_fireflies - 1):
            # Take the NN solution and perturb it
            perturbed = nn_solution.copy()
            # Apply random swaps
            swaps = random.randint(5, 15)
            for _ in range(swaps):
                i, j = random.sample(range(self.dim), 2)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            self.fireflies.append(perturbed)
        
        # Calculate intensity (inverse of distance)
        self.intensities = np.zeros(self.num_fireflies)
        
        # Use parallel processing if enabled
        if self.use_parallel and self.num_fireflies >= 10:
            with multiprocessing.Pool(self.num_cores) as pool:
                self.intensities = np.array(pool.map(self._calc_intensity, self.fireflies))
        else:
            for i, firefly in enumerate(self.fireflies):
                self.intensities[i] = self._calc_intensity(firefly)
    
    def _calc_intensity(self, route):
        """Helper function to calculate intensity of a route."""
        return 1.0 / self._fast_route_distance(route)
    
    def _fast_route_distance(self, route):
        """Faster computation of route distance."""
        # Check if we've calculated this route before
        route_tuple = tuple(route)
        if route_tuple in self.visited_edges:
            return self.visited_edges[route_tuple]
        
        total = 0
        for i in range(self.dim - 1):
            total += self.distance_matrix[route[i]][route[i+1]]
        total += self.distance_matrix[route[-1]][route[0]]  # Return to start
        
        # Cache the result
        self.visited_edges[route_tuple] = total
        return total
    
    def _get_nn_solution(self):
        """Get a nearest neighbor solution with 2-opt improvement."""
        # Start from a random city
        start_city = random.randrange(self.dim)
        
        # Build tour with nearest neighbor heuristic
        unvisited = set(range(self.dim))
        route = [start_city]
        unvisited.remove(start_city)
        
        while unvisited:
            current = route[-1]
            # Find nearest unvisited city
            next_city = min(unvisited, key=lambda city: self.distance_matrix[current][city])
            route.append(next_city)
            unvisited.remove(next_city)
        
        # Apply a quick 2-opt improvement
        improved = True
        while improved:
            improved = False
            for i in range(1, self.dim - 1):
                for j in range(i + 1, self.dim):
                    if j - i == 1:
                        continue  # Skip adjacent edges
                    
                    # Calculate the change in distance
                    a, b = route[i-1], route[i]
                    c, d = route[j], route[(j+1) % self.dim]
                    
                    current_dist = self.distance_matrix[a][b] + self.distance_matrix[c][d]
                    new_dist = self.distance_matrix[a][c] + self.distance_matrix[b][d]
                    
                    if new_dist < current_dist:
                        # Reverse the segment
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                
                if improved:
                    break
        
        return route
    
    def distance_between_fireflies(self, firefly1, firefly2):
        """Simple and fast distance metric between fireflies."""
        # Just count differences in positions for speed
        return sum(1 for i, j in zip(firefly1, firefly2) if i != j)
    
    def move_firefly(self, i, j):
        """Simplified movement for faster execution."""
        # Calculate attraction
        r = self.distance_between_fireflies(self.fireflies[i], self.fireflies[j])
        beta = self.beta_0 * np.exp(-self.gamma * r)
        
        if beta > random.random() * 0.5:  # More likely to move
            # Create a new solution
            new_firefly = self.fireflies[i].copy()
            
            # Apply a faster movement: simple segment reversal
            if random.random() < 0.8:
                # Select a random segment to reverse
                a, b = sorted(random.sample(range(self.dim), 2))
                if b - a > 1:  # Only if segment has at least 2 cities
                    new_firefly[a:b+1] = reversed(new_firefly[a:b+1])
            else:
                # Swap two cities
                a, b = random.sample(range(self.dim), 2)
                new_firefly[a], new_firefly[b] = new_firefly[b], new_firefly[a]
            
            # Update if better
            new_intensity = self._calc_intensity(new_firefly)
            if new_intensity > self.intensities[i]:
                self.fireflies[i] = new_firefly
                self.intensities[i] = new_intensity
    
    def process_firefly_batch(self, firefly_batch):
        """Process a batch of fireflies in parallel."""
        results = []
        
        for i in firefly_batch:
            current_firefly = self.fireflies[i].copy()
            current_intensity = self.intensities[i]
            
            # Compare to a few other fireflies (not all)
            brighter_fireflies = [j for j in range(self.num_fireflies) 
                                  if self.intensities[j] > current_intensity]
            
            if brighter_fireflies:
                # Sample a subset to speed up computation
                sampled_fireflies = random.sample(
                    brighter_fireflies, 
                    min(len(brighter_fireflies), self.max_comparisons)
                )
                
                for j in sampled_fireflies:
                    # Calculate attraction
                    r = self.distance_between_fireflies(current_firefly, self.fireflies[j])
                    beta = self.beta_0 * np.exp(-self.gamma * r)
                    
                    if beta > random.random() * 0.5:
                        # Apply movement
                        if random.random() < 0.8:
                            # Reverse a segment
                            a, b = sorted(random.sample(range(self.dim), 2))
                            if b - a > 1:
                                current_firefly[a:b+1] = reversed(current_firefly[a:b+1])
                        else:
                            # Swap two cities
                            a, b = random.sample(range(self.dim), 2)
                            current_firefly[a], current_firefly[b] = current_firefly[b], current_firefly[a]
                        
                        # Recalculate intensity
                        new_intensity = self._calc_intensity(current_firefly)
                        if new_intensity > current_intensity:
                            current_intensity = new_intensity
                            break  # Stop after first improvement for speed
            
            results.append((i, current_firefly, current_intensity))
        
        return results
    
    def optimize(self):
        """Run the Firefly Algorithm to optimize the TSP."""
        # Clear cache for route distances
        self.visited_edges = {}
        
        # Initialize the population
        self.initialize_population()
        self.convergence = []
        
        # Initialize best solution
        best_idx = np.argmax(self.intensities)
        self.best_solution = self.fireflies[best_idx].copy()
        self.best_distance = 1.0 / self.intensities[best_idx]
        
        # Progress bar only for sequential processing
        iterations_range = tqdm(range(self.max_iterations), desc="Firefly Optimization") if not self.use_parallel else range(self.max_iterations)
        
        # Main optimization loop
        for iteration in iterations_range:
            # Use parallel processing if enabled
            if self.use_parallel and self.num_fireflies > 10:
                # Split fireflies into batches
                batch_size = max(1, self.num_fireflies // self.num_cores)
                firefly_batches = [list(range(i, min(i + batch_size, self.num_fireflies))) 
                                  for i in range(0, self.num_fireflies, batch_size)]
                
                # Process batches in parallel
                with multiprocessing.Pool(self.num_cores) as pool:
                    batch_results = pool.map(self.process_firefly_batch, firefly_batches)
                
                # Update fireflies with results
                for batch in batch_results:
                    for i, new_firefly, new_intensity in batch:
                        self.fireflies[i] = new_firefly
                        self.intensities[i] = new_intensity
            else:
                # Sequential processing
                for i in range(self.num_fireflies):
                    # Sample a subset of brighter fireflies
                    brighter_fireflies = [j for j in range(self.num_fireflies) 
                                          if self.intensities[j] > self.intensities[i]]
                    
                    if brighter_fireflies:
                        # Sample a subset to speed up computation
                        sampled = random.sample(
                            brighter_fireflies, 
                            min(len(brighter_fireflies), self.max_comparisons)
                        )
                        
                        for j in sampled:
                            self.move_firefly(i, j)
            
            # Update best solution
            best_idx = np.argmax(self.intensities)
            current_best_distance = 1.0 / self.intensities[best_idx]
            
            if current_best_distance < self.best_distance:
                self.best_solution = self.fireflies[best_idx].copy()
                self.best_distance = current_best_distance
            
            # Record convergence
            self.convergence.append(self.best_distance)
            
            # Apply 2-opt to best solution every 10 iterations
            if iteration % 10 == 0 or iteration == self.max_iterations - 1:
                improved = self._quick_2opt(self.best_solution)
                improved_distance = self._fast_route_distance(improved)
                
                if improved_distance < self.best_distance:
                    self.best_solution = improved
                    self.best_distance = improved_distance
            
            # Dynamic adjustment of parameters
            self.alpha *= 0.97
            
            # Early termination if no improvement for 15 iterations
            if len(self.convergence) > 15 and all(abs(self.convergence[-1] - self.convergence[-i-1]) < 0.01 for i in range(15)):
                print(f"Early stopping at iteration {iteration+1} due to convergence")
                break
        
        # Final quick optimization
        self.best_solution = self._quick_2opt(self.best_solution)
        self.best_distance = self._fast_route_distance(self.best_solution)
        
        # Clear cache to save memory
        self.visited_edges = {}
        
        return self.best_solution, self.best_distance, self.convergence
    
    def _quick_2opt(self, route, max_passes=2):
        """Faster 2-opt implementation for final improvement."""
        improved = True
        best_route = route.copy()
        pass_count = 0
        
        while improved and pass_count < max_passes:
            improved = False
            pass_count += 1
            
            # Try a limited number of random 2-opt moves
            for _ in range(30):  # Limited number of attempts
                # Choose random edges
                i, j = sorted(random.sample(range(1, self.dim), 2))
                if j - i <= 1:
                    continue
                
                # Calculate the change in distance
                a, b = best_route[i-1], best_route[i]
                c, d = best_route[j], best_route[(j+1) % self.dim]
                
                current_dist = self.distance_matrix[a][b] + self.distance_matrix[c][d]
                new_dist = self.distance_matrix[a][c] + self.distance_matrix[b][d]
                
                if new_dist < current_dist:
                    # Reverse the segment
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    improved = True
                    break
        
        return best_route
