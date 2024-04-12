import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100,
                'F_base': 0.5,
                'CR': 0.9,
                'alpha': 0.5,
                'p': 0.2,
                'gamma': 0.05,  # New parameter for controlling diversity
                'beta': 0.001,
                'memory_size': 5  # Memory for adaptive F
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.memory_F = np.full(self.params['memory_size'], self.params['F_base'])

    def evaluate(self, solutions):
        fitness = np.asarray([self.func(ind) for ind in solutions])
        self.evals += len(solutions)
        return fitness

    def adaptive_F(self):
        # Adapt F based on historical successful F values
        return np.random.choice(self.memory_F)

    def update_memory_F(self, F):
        # Update memory with new successful F value
        self.memory_F = np.roll(self.memory_F, -1)
        self.memory_F[-1] = F

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        CR = self.params.get('CR', 0.9)
        alpha = self.params.get('alpha', 0.5)
        p = self.params.get('p', 0.2)
        gamma = self.params.get('gamma', 0.05)
        beta = self.params.get('beta', 0.001)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)
        
        best_index = np.argmin(fitness)
        best = pop[best_index]
        best_fitness = fitness[best_index]

        while self.evals < self.max_evals:
            for i in range(pop_size):
                F = self.adaptive_F()
                # Mutation: DE/rand/1
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover: bin
                cross_points = np.random.rand(self.dim) <= CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Local Search with probability p
                if np.random.rand() < p:
                    local_search_vector = best + alpha * (np.random.rand(self.dim) - 0.5) * (self.bounds[1] - self.bounds[0])
                    trial = np.clip(local_search_vector, self.bounds[0], self.bounds[1])

                # Selection
                trial_fitness = self.func(trial)
                self.evals += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    self.update_memory_F(F)  # Update memory with successful F
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

                # Introduce diversity if the population converges
                if np.random.rand() < gamma:
                    pop[i] = np.random.rand(self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                    fitness[i] = self.func(pop[i])
                    self.evals += 1

                # Adaptive penalty for stagnation
                if self.evals % (10 * pop_size) == 0:
                    fitness += beta * np.abs(fitness - best_fitness)

                if self.evals >= self.max_evals:
                    break

        return best_fitness

