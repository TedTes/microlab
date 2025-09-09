import numpy as np

class CoinTossSimulator:
    def single_experiment(self, n_tosses:int,  bias:float = 0.5) -> dict:
        random_values = np.random.random(n_tosses)
        tosses = random_values < bias
        heads_count = np.sum(tosses)
      
        proportion = heads_count / n_tosses

        return {
            'tosses': tosses,
            'heads_count': heads_count, 
            'proportion': proportion,
            'bias': bias,
            'n_tosses': n_tosses
        }

    def demonstrate_law_of_large_numbers(self):
        print("Law of Large Numbers Demo:")
        print("Sample Size -> Proportion -> Error")
        for bias in [0.5, 0.7]:
            print(f'for bias {bias}')
            for n_tosses in [10, 50, 100, 500, 1000, 5000,10000,20000]:
                result = self.single_experiment(n_tosses,bias)
                error = abs(result['proportion'] - bias)
                print(f"{n_tosses:4d} -> {result['proportion']:.3f} -> {error:.3f}")
    def convergence_study(self, max_tosses:int=1000, bias: float = 0.5, step_size:int=10)  -> dict:
         """
                Study how sample proportion converges to true bias.

            Args:
                max_tosses: Maximum number of tosses to simulate  
                bias: True probability of heads
                step_size: How often to record the running proportion
                
            Returns:
                dict with 'sample_sizes', 'proportions', 'true_bias'
         """
         sample_sizes = []
         proportions = []
         all_tosses = np.random.random(max_tosses) < bias
         for n in range(step_size, max_tosses + 1, step_size):
            running_proportion = np.sum(all_tosses[:n]) / n
            sample_sizes.append(n)
            proportions.append(running_proportion)
         return {
                'sample_sizes': sample_sizes,
                'proportions': proportions,
                'true_bias': bias
          }
    
    def central_limit_demo(self, sample_size:int = 100,  n_experiments:int = 1000, bias:float=0.5) -> dict: 
        """
            Demonstrate Central Limit Theorem with coin tosses.
            
            Args:
                sample_size: Number of tosses per experiment
                n_experiments: Number of experiments to run  
                bias: Coin bias
                
            Returns:
                dict with sample means and theoretical predictions
        """

        sample_means = []
        for i in range(n_experiments):
            experiment_result = self.single_experiment(sample_size,bias)
            proportion = experiment_result['proportion']
            sample_means.append(proportion)

        theoretical_mean = bias
        theoretical_std = np.sqrt(bias * (1 - bias) / sample_size)

        return {
            'sample_means': sample_means,
            'theoretical_mean': theoretical_mean,
            'theoretical_std': theoretical_std,
            'sample_size': sample_size,
            'n_experiments': n_experiments
        }

    
simulator = CoinTossSimulator()
# simulator.demonstrate_law_of_large_numbers()
# simulator.single_experiment(100)
# result = simulator.convergence_study(max_tosses=100, bias=0.7, step_size=10)
# print("Sample sizes:", result['sample_sizes'])
# print("Proportions:", result['proportions'])


result = simulator.central_limit_demo(sample_size=100, n_experiments=10, bias=0.7)
print("Sample means:", result['sample_means'])
print("Should center around:", result['theoretical_mean'])
print("Standard deviation:", result['theoretical_std'])

