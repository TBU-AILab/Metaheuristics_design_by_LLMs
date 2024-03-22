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
            'learning_phase': True,  # Introduce a learning phase for parameter adaptation
            'learning_rate': 0.01,  # Learning rate for parameter adaptation
            'mutation_strategy': 'DE/rand/1/bin',  # Default mutation strategy
            'hybridization': True,  # Enable hybridization with local search
            'local_search_rate': 0.05,  # Rate at which local search is applied
            'local_search_step': 0.1,  # Step size for local search
        }

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F_base = self.params.get('F', 0.5)
        Cr_base = self.params.get('CR', 0.7)
        adaptive = self.params.get('adaptive', False)
        learning_phase = self.params.get('learning_phase', False)
        learning_rate = self.params.get('learning_rate', 0.01)
        hybridization = self.params.get('hybridization', False)
        local_search_rate = self.params.get('local_search_rate', 0.05)
        local_search_step = self.params.get('local_search_step', 0.1)

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
            if adaptive:
                # Adapt F and Cr using a simple learning rule
                F_base = F_base + learning_rate * (np.random.rand() - 0.5)
                Cr_base = Cr_base + learning_rate * (np.random.rand() - 0.5)
                F_base = np.clip(F_base, 0.1, 1)
                Cr_base = np.clip(Cr_base, 0, 1)

            for i in range(pop_size):
                if hybridization and np.random.rand() < local_search_rate:
                    # Perform local search on a randomly selected individual
                    local_search_idx = np.random.randint(pop_size)
                    for d in range(self.dim):
                        trial = np.copy(pop[local_search_idx])
                        trial[d] += local_search_step * (2 * np.random.rand() - 1)
                        trial = np.clip(trial, self.bounds[0], self.bounds[1])
                        trial_fitness = self.func(trial)
                        evals += 1
                        if trial_fitness < fitness[local_search_idx]:
                            pop[local_search_idx] = trial
                            fitness[local_search_idx] = trial_fitness
                            if trial_fitness < best['fitness']:
                                best['fitness'] = trial_fitness
                                best['params'] = trial
                                best['gen'] = evals // pop_size
                                best['eval_num'] = evals

                # DE Mutation and Crossover
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F_base * (b - c), self.bounds[0], self.bounds[1])
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

