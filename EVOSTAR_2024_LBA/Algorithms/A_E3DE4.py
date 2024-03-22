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
            'memory': True,  # Enable memory for storing best solutions
            'memory_size': 5,  # Size of the memory
        }

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F_base = self.params.get('F', 0.5)
        Cr_base = self.params.get('CR', 0.7)
        adaptive = self.params.get('adaptive', False)
        memory_enabled = self.params.get('memory', False)
        memory_size = self.params.get('memory_size', 5)

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

        # Initialize memory
        memory = {'params': [], 'fitness': []}

        while evals < self.max_evals:
            if adaptive:
                # Dynamically adjust F and Cr
                F_base = np.clip(F_base + 0.01 * np.sin(evals / 1000), 0.1, 1)
                Cr_base = np.clip(Cr_base + 0.01 * np.cos(evals / 1000), 0, 1)

            for i in range(pop_size):
                # Mutation and Crossover
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

                        # Update memory with new best
                        if memory_enabled:
                            if len(memory['fitness']) < memory_size:
                                memory['params'].append(trial)
                                memory['fitness'].append(trial_fitness)
                            else:
                                # Replace the worst in memory if the new best is better
                                worst_index = np.argmax(memory['fitness'])
                                if trial_fitness < memory['fitness'][worst_index]:
                                    memory['params'][worst_index] = trial
                                    memory['fitness'][worst_index] = trial_fitness

                # Check if memory should be used to replace a random individual
                if memory_enabled and len(memory['fitness']) > 0 and np.random.rand() < 0.1:
                    replace_index = np.random.randint(0, pop_size)
                    memory_index = np.random.randint(0, len(memory['fitness']))
                    pop[replace_index] = memory['params'][memory_index]
                    fitness[replace_index] = memory['fitness'][memory_index]

                if evals >= self.max_evals:
                    break

        return best

