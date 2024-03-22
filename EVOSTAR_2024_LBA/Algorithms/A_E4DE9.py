import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.params = {
            'pop_size': 20,
            'F': 0.8,
            'CR': 0.9,
            'adaptive': True,
            'hybridization': True,
            'local_search_rate': 0.05,
            'local_search_step': 0.1,
            'diversity_enhancement': True,
            'diversity_threshold': 0.1,
            'novelty_search': True,
            'novelty_threshold': 0.05,
            'archive_size': 50,
            'strategy_adaptation': True,  # New component for strategy adaptation
            'strategy_pool': ['DE/rand/1/bin', 'DE/best/2/bin', 'DE/current-to-best/1/bin'],  # Pool of strategies
            'strategy_change_freq': 1000,  # Frequency of strategy change
        }

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F_base = self.params.get('F', 0.5)
        Cr_base = self.params.get('CR', 0.7)
        adaptive = self.params.get('adaptive', False)
        hybridization = self.params.get('hybridization', False)
        local_search_rate = self.params.get('local_search_rate', 0.05)
        local_search_step = self.params.get('local_search_step', 0.1)
        diversity_enhancement = self.params.get('diversity_enhancement', False)
        diversity_threshold = self.params.get('diversity_threshold', 0.1)
        novelty_search = self.params.get('novelty_search', False)
        novelty_threshold = self.params.get('novelty_threshold', 0.05)
        archive_size = self.params.get('archive_size', 50)
        strategy_adaptation = self.params.get('strategy_adaptation', False)
        strategy_pool = self.params.get('strategy_pool', ['DE/rand/1/bin', 'DE/best/2/bin', 'DE/current-to-best/1/bin'])
        strategy_change_freq = self.params.get('strategy_change_freq', 1000)

        # Initialize population and archive for novelty search
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size
        archive = pop[np.random.choice(range(pop_size), size=min(archive_size, pop_size), replace=False)]
        current_strategy = strategy_pool[0]

        best_index = np.argmin(fitness)
        best = {
            'params': pop[best_index],
            'fitness': fitness[best_index],
            'gen': 0,
            'eval_num': evals
        }

        while evals < self.max_evals:
            if strategy_adaptation and evals % strategy_change_freq == 0:
                # Change strategy periodically
                current_strategy = np.random.choice(strategy_pool)

            for i in range(pop_size):
                if hybridization and np.random.rand() < local_search_rate:
                    # Perform local search
                    trial = pop[i] + local_search_step * (2 * np.random.rand(self.dim) - 1)
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])
                    trial_fitness = self.func(trial)
                    evals += 1
                    if trial_fitness < fitness[i]:
                        pop[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best['fitness']:
                            best['fitness'] = trial_fitness
                            best['params'] = trial
                            best['gen'] = evals // pop_size
                            best['eval_num'] = evals

                # Apply current strategy for DE Mutation and Crossover
                if current_strategy == 'DE/rand/1/bin':
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F_base * (b - c), self.bounds[0], self.bounds[1])
                elif current_strategy == 'DE/best/2/bin':
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b = pop[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(best['params'] + F_base * (a + b - 2 * pop[i]), self.bounds[0], self.bounds[1])
                elif current_strategy == 'DE/current-to-best/1/bin':
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a = pop[np.random.choice(idxs, 1, replace=False)][0]
                    mutant = np.clip(pop[i] + F_base * (best['params'] - pop[i] + a - pop[i]), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) <= Cr_base
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best['fitness']:
                        best['fitness'] = trial_fitness
                        best['params'] = trial
                        best['gen'] = evals // pop_size
                        best['eval_num'] = evals

                if evals >= self.max_evals:
                    break

        return best

