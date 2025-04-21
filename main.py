import os
import time
from src.tsp import TSPProblem
from src.firefly import FireflyAlgorithm
from src.utils import visualize_route, plot_convergence, save_results
import multiprocessing

def main():
    """Main function to run the Firefly Algorithm on a TSP instance."""
    print("Traveling Salesman Problem Solver using Firefly Algorithm")
    print("-" * 60)
    
    # Load the Berlin52 dataset
    berlin_path = os.path.join("data", "berlin52.tsp")
    problem = TSPProblem(file_path=berlin_path)
    print(f"Loaded problem: {problem.name} with {problem.dimension} cities")
    
    # Configure for speed with reasonable quality
    num_fireflies = 40  # Reduced from 80
    alpha = 0.2
    beta_0 = 1.0
    gamma = 0.01
    max_iterations = 100  # Reduced from 500
    
    # Use multiprocessing if available
    num_cores = multiprocessing.cpu_count()
    use_parallel = True
    
    algorithm_params = {
        "num_fireflies": num_fireflies,
        "alpha": alpha,
        "beta_0": beta_0,
        "gamma": gamma,
        "max_iterations": max_iterations,
        "parallel_processing": use_parallel,
        "num_cores": num_cores
    }
    
    print("\nFast Execution Parameters:")
    for key, value in algorithm_params.items():
        print(f"  {key}: {value}")
    
    # Initialize and run the Firefly Algorithm
    print("\nStarting optimization...")
    firefly_algo = FireflyAlgorithm(
        problem, 
        num_fireflies=num_fireflies,
        alpha=alpha,
        beta_0=beta_0,
        gamma=gamma,
        max_iterations=max_iterations,
        use_parallel=use_parallel,
        num_cores=num_cores
    )
    
    start_time = time.time()
    best_route, best_distance, convergence = firefly_algo.optimize()
    execution_time = time.time() - start_time
    
    print(f"\nOptimization completed in {execution_time:.2f} seconds")
    print(f"Best distance found: {best_distance:.4f}")
    print(f"Optimal distance for Berlin52: 7,542 units")
    print(f"Gap to optimal: {((best_distance/7542) - 1) * 100:.2f}%")
    
    # Save and visualize results
    results_dir = save_results(
        problem, 
        best_route, 
        best_distance, 
        convergence, 
        algorithm_params, 
        execution_time
    )
    
    print(f"\nResults saved to: {results_dir}")
    
    # Save sample images for README documentation
    if not os.path.exists(os.path.join("results", "sample_route.png")):
        sample_path = os.path.join("results", "sample_route.png")
        visualize_route(problem, best_route, save_path=sample_path, show=False)
        print(f"Saved sample route visualization to {sample_path}")
        
    if not os.path.exists(os.path.join("results", "sample_convergence.png")):
        sample_path = os.path.join("results", "sample_convergence.png")
        plot_convergence(convergence, save_path=sample_path, show=False)
        print(f"Saved sample convergence plot to {sample_path}")
    
    # Visualize the solution
    print("\nVisualizing best route...")
    visualize_route(problem, best_route)
    
    print("\nPlotting convergence history...")
    plot_convergence(convergence)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
