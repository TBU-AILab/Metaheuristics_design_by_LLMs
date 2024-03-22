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
            'p_best': 0.2,  # Percentage for p-best strategy
            'mutation_strategy': 'current-to-pbest/1',  # New mutation strategy
        }

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F = self.params.get('F', 0.5)
        Cr = self.params.get('CR', 0.7)
        p_best = self.params.get('p_best', 0.2)
        mutation_strategy = self.params.get('mutation_strategy', 'rand/1')

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size

        best_index = np.argmin(fitness)
        best = {
            'params': pop[best_index],
            'fitness': fitness[best_index],
            'gen': 0,
            'eval_num': evals
        }

        while evals < self.max_evals:
            sorted_indices = np.argsort(fitness)
            p_best_size = int(pop_size * p_best)
            p_best_pop = pop[sorted_indices[:p_best_size]]

            for i in range(pop_size):
                # Mutation
                if mutation_strategy == 'current-to-pbest/1':
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    p_best_idx = np.random.choice(range(p_best_size))
                    a = p_best_pop[p_best_idx]
                    b, c = pop[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(pop[i] + F * (a - pop[i]) + F * (b - c), self.bounds[0], self.bounds[1])
                else:  # Default 'rand/1' strategy
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                evals += 1
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness <= best['fitness']:
                        best['fitness'] = trial_fitness
                        best['params'] = trial
                        best['gen'] = evals // pop_size
                        best['eval_num'] = evals

                if evals >= self.max_evals:
                    break

        return best

