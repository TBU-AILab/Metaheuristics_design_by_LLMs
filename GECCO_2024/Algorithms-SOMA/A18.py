import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                "pop_size": 150,  # Further increased population size
                "step_size": 0.05,  # Reduced step size for finer exploration
                "prt": 0.5,  # Increased perturbation rate for enhanced diversity
                "path_length": 3.5,  # Slightly increased path length
                "mutation_rate": 0.1,  # Slightly increased mutation rate
                "crossover_rate": 0.95,  # Increased crossover rate for more genetic diversity
                "elite_rate": 0.1,  # Introduced elitism to preserve best solutions
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

    def run(self):
        pop_size = self.params.get('pop_size')
        step_size = self.params.get('step_size')
        prt = self.params.get('prt')
        path_length = self.params.get('path_length')
        mutation_rate = self.params.get('mutation_rate')
        crossover_rate = self.params.get('crossover_rate')
        elite_rate = self.params.get('elite_rate')

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        pop_cf = np.array([self.func(ind) for ind in pop])
        evals = pop_size
        elite_size = int(pop_size * elite_rate)

        while evals < self.max_evals:
            # Sort population based on fitness (cost function values)
            sorted_indices = np.argsort(pop_cf)
            pop = pop[sorted_indices]
            pop_cf = pop_cf[sorted_indices]

            # Elitism: Copy elite individuals directly to the new population
            new_pop = np.empty_like(pop)
            new_pop_cf = np.empty_like(pop_cf)
            new_pop[:elite_size] = pop[:elite_size]
            new_pop_cf[:elite_size] = pop_cf[:elite_size]

            for i in range(elite_size, pop_size):
                if np.random.rand() < crossover_rate:
                    # Select another individual for crossover
                    j = np.random.randint(pop_size)
                    while j == i:
                        j = np.random.randint(pop_size)
                    # Crossover
                    crossover_point = np.random.randint(0, self.dim)
                    new_pop[i, :crossover_point] = pop[j, :crossover_point]

                # Mutation
                if np.random.rand() < mutation_rate:
                    mutation_point = np.random.randint(self.dim)
                    new_pop[i, mutation_point] += np.random.normal(0, 1)

                # Ensure new individual is within bounds
                new_pop[i] = np.clip(new_pop[i], self.bounds[0], self.bounds[1])
                new_pop_cf[i] = self.func(new_pop[i])
                evals += 1

                if evals >= self.max_evals:
                    break

            # Update population
            pop = np.copy(new_pop)
            pop_cf = np.copy(new_pop_cf)

        best_index = np.argmin(pop_cf)
        best = pop[best_index]
        best_cf = pop_cf[best_index]

        return best, best_cf

