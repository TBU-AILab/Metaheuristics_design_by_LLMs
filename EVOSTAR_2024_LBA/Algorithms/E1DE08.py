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
            'diversity_enhancement': True,  # New component for enhancing diversity
            'diversity_threshold': 0.1,  # Threshold for triggering diversity enhancement
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
                F_base = np.clip(F_base + 0.01 * (np.random.rand() - 0.5), 0.1, 1)
                Cr_base = np.clip(Cr_base + 0.01 * (np.random.rand() - 0.5), 0, 1)

            if diversity_enhancement:
                # Measure diversity
                mean_pop = np.mean(pop, axis=0)
                diversity = np.mean(np.linalg.norm(pop - mean_pop, axis=1))
                if diversity < diversity_threshold * np.linalg.norm(self.bounds[1] - self.bounds[0]):
                    # Enhance diversity by random perturbation
                    perturbation = np.random.randn(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) * 0.1
                    pop = np.clip(pop + perturbation, self.bounds[0], self.bounds[1])
                    fitness = np.asarray([self.func(ind) for ind in pop])
                    evals += pop_size

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

