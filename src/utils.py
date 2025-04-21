import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

def visualize_route(problem, route, save_path=None, show=True):
    """
    Visualize a TSP route on a plot.
    
    Parameters:
    -----------
    problem : TSPProblem
        The TSP problem instance
    route : list
        List of city indices representing a route
    save_path : str, optional
        Path to save the visualization
    show : bool, optional
        Whether to display the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot cities
    x = [problem.coordinates[i][0] for i in range(problem.dimension)]
    y = [problem.coordinates[i][1] for i in range(problem.dimension)]
    plt.scatter(x, y, c='blue', s=40)
    
    # Plot route
    for i in range(len(route) - 1):
        plt.plot([problem.coordinates[route[i]][0], problem.coordinates[route[i+1]][0]],
                 [problem.coordinates[route[i]][1], problem.coordinates[route[i+1]][1]], 'k-', alpha=0.6)
    
    # Connect last city to first city
    plt.plot([problem.coordinates[route[-1]][0], problem.coordinates[route[0]][0]],
             [problem.coordinates[route[-1]][1], problem.coordinates[route[0]][1]], 'k-', alpha=0.6)
    
    # Add details
    distance = problem.get_route_distance(route)
    plt.title(f"TSP Solution for {problem.name}\nTotal Distance: {distance:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_convergence(convergence_history, save_path=None, show=True):
    """
    Plot the convergence history of the algorithm.
    
    Parameters:
    -----------
    convergence_history : list
        List of best distances at each iteration
    save_path : str, optional
        Path to save the plot
    show : bool, optional
        Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(convergence_history, 'b-')
    plt.title("Convergence of Firefly Algorithm")
    plt.xlabel("Iteration")
    plt.ylabel("Best Distance")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def save_results(problem, route, distance, convergence, algorithm_params, execution_time, results_dir="results"):
    """
    Save the results of the optimization run.
    
    Parameters:
    -----------
    problem : TSPProblem
        The TSP problem instance
    route : list
        The optimal route found
    distance : float
        The distance of the optimal route
    convergence : list
        Convergence history
    algorithm_params : dict
        Parameters of the algorithm
    execution_time : float
        Execution time in seconds
    results_dir : str
        Directory to save results
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"{problem.name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save route visualization
    visualize_route(problem, route, save_path=os.path.join(run_dir, "route.png"), show=False)
    
    # Save convergence plot
    plot_convergence(convergence, save_path=os.path.join(run_dir, "convergence.png"), show=False)
    
    # Save route and metrics
    with open(os.path.join(run_dir, "results.txt"), 'w') as f:
        f.write(f"Problem: {problem.name}\n")
        f.write(f"Dimension: {problem.dimension}\n")
        f.write(f"Best Distance: {distance:.4f}\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
        
        f.write("Algorithm Parameters:\n")
        for key, value in algorithm_params.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nBest Route:\n")
        f.write(" -> ".join(map(str, route)) + f" -> {route[0]}\n")
    
    # Save raw data
    np.savetxt(os.path.join(run_dir, "convergence.csv"), convergence)
    np.savetxt(os.path.join(run_dir, "route.csv"), route, fmt='%d')
    
    return run_dir
