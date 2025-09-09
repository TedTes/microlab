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


result = CoinTossSimulator().single_experiment(100)
print(result)
