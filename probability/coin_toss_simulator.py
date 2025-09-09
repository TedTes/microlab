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

simulator = CoinTossSimulator()
simulator.demonstrate_law_of_large_numbers()
