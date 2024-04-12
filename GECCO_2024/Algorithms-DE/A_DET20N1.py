import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100,
                'F': 0.5,
                'CR': 0.9,
                'alpha': 0.3,
                'p': 0.1
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0

    def evaluate(self, solutions):
        fitness = np.asarray([self.func(ind) for ind in solutions])
        self.evals += len(solutions)
        return fitness

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        F = self.params.get('F', 0.5)
        Cr = self.params.get('CR', 0.9)
        alpha = self.params.get('alpha', 0.3)
        p = self.params.get('p', 0.1)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)
        
        best_index = np.argmin(fitness)
        best = pop[best_index]
        best_fitness = fitness[best_index]

        while self.evals < self.max_evals:
            for i in range(pop_size):
                # Mutation: DE/rand/1
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover: bin
                cross_points = np.random.rand(self.dim) <= Cr
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
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

                if self.evals >= self.max_evals:
                    break

        return best_fitness

