from env_creator import qsimpy_env_creator
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import List, Dict
import random

class ACOSolution:
    def __init__(self, env, num_episodes=100, num_ants=10, num_iterations=50):
        self.env = env
        self.num_episodes = num_episodes
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        
        # ACO parameters
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0   # Heuristic importance
        self.evaporation_rate = 0.1
        self.initial_pheromone = 1.0
        
        # Results storage
        self.results = []
        
    def run(self):
        self.results = []
        
        for episode in range(self.num_episodes):
            # Initialize temporary results for this episode
            episode_result = {
                "total_completion_time": 0.0,
                "rescheduling_count": 0.0
            }
            
            # Reset environment
            self.env.reset()
            self.env.setup_quantum_resources()
            
            # Initialize pheromone matrix for this episode
            pheromone_matrix = np.ones((self.env.n_qtasks, self.env.n_qnodes)) * self.initial_pheromone
            
            terminated = False
            current_task = 0
            
            while not terminated:
                # Run ACO for current task
                best_node = self._run_aco_iteration(current_task, pheromone_matrix)
                
                # Take action in environment
                obs, reward, terminated, done, info = self.env.step(best_node)
                
                if reward > 0:
                    # Update results if task was successfully scheduled
                    episode_result["total_completion_time"] += (
                        info["scheduled_qtask"].waiting_time + 
                        info["scheduled_qtask"].execution_time
                    )
                    episode_result["rescheduling_count"] += info["scheduled_qtask"].rescheduling_count
                    current_task += 1
                    
                    # Update pheromone for successful allocation
                    self._update_pheromone(pheromone_matrix, current_task-1, best_node, reward)
            
            self.env.qsp_env.run()
            self.results.append(episode_result)
            
        # Save results
        self._save_to_csv()
    
    def _run_aco_iteration(self, task_id: int, pheromone_matrix: np.ndarray) -> int:
        best_node = None
        best_score = float('-inf')
        
        for _ in range(self.num_iterations):
            for ant in range(self.num_ants):
                # Calculate node selection probabilities
                probabilities = self._calculate_probabilities(task_id, pheromone_matrix)
                
                # Select node based on probabilities
                node = np.random.choice(len(probabilities), p=probabilities)
                
                # Evaluate node selection
                score = self._evaluate_node(task_id, node)
                
                if score > best_score:
                    best_score = score
                    best_node = node
        
        return best_node
    
    def _calculate_probabilities(self, task_id: int, pheromone_matrix: np.ndarray) -> np.ndarray:
        # Get pheromone values for current task
        pheromone = pheromone_matrix[task_id]
        
        # Calculate heuristic values based on node availability
        heuristic = np.array([1.0 / (node.next_available_time + 1e-10) 
                            for node in self.env.qnodes])
        
        # Combined probability calculation
        probabilities = (pheromone ** self.alpha) * (heuristic ** self.beta)
        return probabilities / np.sum(probabilities)
    
    def _evaluate_node(self, task_id: int, node_id: int) -> float:
        # Simple evaluation based on node availability and error rate
        node = self.env.qnodes[node_id]
        score = -node.next_available_time
            
        return score
    
    def _update_pheromone(self, pheromone_matrix: np.ndarray, task_id: int, 
                         node_id: int, reward: float):
        # Evaporation
        pheromone_matrix *= (1 - self.evaporation_rate)
        
        # Deposit new pheromone
        deposit = reward * 10  # Scale reward for pheromone deposit
        pheromone_matrix[task_id, node_id] += deposit
        
        # Normalize pheromone values
        pheromone_matrix = np.clip(pheromone_matrix, 0.1, 5.0)
    
    def _save_to_csv(self):
        file_name = "./results/aco/aco.csv"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total Completion Time', 'Rescheduling Count'])
            
            for i in range(len(self.results)):
                writer.writerow([
                    i,
                    self.results[i]['total_completion_time'],
                    self.results[i]['rescheduling_count']
                ])
        print("ACO results saved to " + file_name)


class HeuristicSolutions:
    def __init__(self, env, num_episodes=100):

        # Initialize the environment
        self.env = env
        self.num_episodes = num_episodes

        # Initialize the results of heuristic solutions
        self.results = []
        # Round Robin index for the QNodes. Example: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...]
        self.rr_index = 0
        # Priority index of Greedy solution after sorting the QNodes based on the waiting time
        self.greedy_index = 0

    def run(self, control):
        """
        Run the heuristic solutions for the given algorithm (control).
        Args:
            - control (str): The heuristic algorithm to use. Options: "greedy", "random", "round_robin", "greedy_error"
        """

        self.results = []
        # Reset the subset of QTasks 
        self.env.round = 1

        for _ in range(self.num_episodes):

            # Initialize the temporary array to store the results of the QTasks execution for each episode
            arr_temp = {
                "total_completion_time": 0.0,
                "rescheduling_count": 0.0
            }
            terminated = False

            # Reset the environment and setup the quantum resources
            self.env.reset()
            self.env.setup_quantum_resources()
            self.rr_index = 0

            while not terminated:
                # Get the action with the given control
                if control == "greedy":
                    action = self.greedy(self.greedy_index)
                elif control == "random":
                    action = self.random()
                elif control == "round_robin":
                    action = self.round_robin()
                elif control == "greedy_error":
                    action = self.greedy_error(self.greedy_index)
                
                obs, reward, terminated, done, info = self.env.step(action)
                
                # If the QNode is busy or not satisfied, move to the next priority QNode
                self.greedy_index += 1
                if reward > 0:
                    """Get the results of the QTask execution

                    Values:
                        - Total Completion Time: waiting_time + execution_time
                        - Rescheduling Count: rescheduling_count
                    """
                    # Reset priority index of Greedy solution if QTasks are satisfied
                    self.greedy_index = 0

                    arr_temp["total_completion_time"] += info["scheduled_qtask"].waiting_time + info["scheduled_qtask"].execution_time
                    arr_temp["rescheduling_count"] += info["scheduled_qtask"].rescheduling_count
            self.env.qsp_env.run()
            # Final results of the episode
            self.results.append(arr_temp)

        # Save the results to a CSV file
        self._save_to_csv(control)
                
    def greedy(self, greedy_index):
        # Sort the QNodes based on the next available time (or waiting time) and select the QNode with the smallest waiting time
        greedy_strategy = sorted(self.env.qnodes, key=lambda x: x.next_available_time)
        return self.env.qnodes.index(greedy_strategy[greedy_index])

    def random(self):
        # Randomly select a QNode
        action = self.env.action_space.sample()
        return action
    
    def round_robin(self):
        # Select the QNode based on the Round Robin index
        action = self.rr_index % self.env.n_qnodes
        self.rr_index += 1
        return action
    
    def greedy_error(self, greedy_index, g_error="Readout_assignment_error"):
        # Sort the QNodes based on the next available time (or waiting time) and select the QNode with the 
        # smallest waiting time and smallest error (default is readout_error) in the qnode
    
        greedy_strategy = sorted(self.env.qnodes, key=lambda x: (x.next_available_time, x.error[g_error]))
        return self.env.qnodes.index(greedy_strategy[greedy_index])

    def _save_to_csv(self, control) -> None:
        """
        Save values and episodes to a CSV file.
        """

        file_name = "./results/heuristics2/" 

        if not os.path.exists(file_name):
            os.makedirs(file_name)

        file_name += control + ".csv"
        # Open the CSV file in write mode
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(['Episode', 'Total Completion Time', 'Rescheduling Count'])
            
            # Write the data
            for i in range(len(self.results)):
                writer.writerow([i, self.results[i]['total_completion_time'], self.results[i]['rescheduling_count']])
        print("CSV file saved to " + file_name)

    def _plot_results(self, paths) -> None:
        """
        Plot the results of the episodes.
        """
        for path in paths:
            df1 = pd.read_csv(path['path'])

            plt.plot(df1['Episode'], df1['Total Completion Time'], ".-", color=path['color'], label=path['label'])

            self._summarize_results(df1, path['label'])
        
        plt.ylabel('Total Completion Time')
        plt.xlabel('Evaluation Episode')
        plt.legend(loc=2)
        plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
        plt.show()

    def _summarize_results(self, values, label) -> None:
        """
        Summarize the results of the episodes.
        """
        print("Results Summary for" + label + "solution:")
        print(f"Number of Episodes: {self.num_episodes}")
        print(f"Total Completion Time: {sum(values['Total Completion Time'])}")
        print(f"Average Rescheduling Count: {sum(values['Rescheduling Count']) / self.num_episodes}")



def plot_comparative_results(algorithms: List[Dict]):
    plt.figure(figsize=(12, 6))
    
    for algo in algorithms:
        df = pd.read_csv(algo['path'])
        plt.plot(df['Episode'], df['Total Completion Time'], 
                ".-", color=algo['color'], label=algo['label'])
        
        # Print summary statistics
        print(f"\nResults Summary for {algo['label']}:")
        print(f"Average Completion Time: {df['Total Completion Time'].mean():.2f}")
        print(f"Average Rescheduling Count: {df['Rescheduling Count'].mean():.2f}")
    
    plt.title('Comparison of Task Allocation Strategies')
    plt.ylabel('Total Completion Time')
    plt.xlabel('Evaluation Episode')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))
    
    # Save the plot
    plt.savefig('./results/heuristics2/comparison_plot.png')
    plt.show()

if __name__ == "__main__":

    # Create the QSimPy environment
    env_config={
                "obs_filter": "rescale_-1_1",
                "reward_filter": None,
                "dataset": "qdataset/qsimpyds_1000_sub_26.csv",
            }

    env = qsimpy_env_creator(env_config)

    # Run the heuristic solutions
    heuristics = HeuristicSolutions(env, num_episodes=100)
    heuristics.run("greedy")
    heuristics.run("random")
    heuristics.run("round_robin")
    heuristics.run("greedy_error")

    #Run the ACO 
    aco = ACOSolution(env, num_episodes=100, num_ants=10, num_iterations=50)
    aco.run()

    # Plot the results
    paths = [
        {
            "label": "random",
            "path": "./results/heuristics2/random.csv",
            "color": "red"
        },
        {
            "label": "round robin",
            "path": "./results/heuristics2/round_robin.csv",
            "color": "blue"
        },
        {
            "label": "greedy",
            "path": "./results/heuristics2/greedy.csv",
            "color": "black"
        },
        {
            "label": "greedy_error",
            "path": "./results/heuristics2/greedy_error.csv",
            "color": "green"
        },
        {
            "label": "ACO", 
            "path": "./results/heuristics2/aco.csv", 
            "color": "purple"
        }

    ]
    #heuristics._plot_results(paths)
    plot_comparative_results(paths)
