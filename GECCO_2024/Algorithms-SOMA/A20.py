import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                "pop_size": 1200,  # Expanding population size for even broader exploration
                "step_size": 0.0000005,  # Ultra-fine step size for the utmost precision
                "prt": 1.0,  # Perturbation at maximum for comprehensive exploration
                "path_length": 15.0,  # Further extended path length for exhaustive searches
                "mutation_rate": 0.9,  # Maximizing mutation rate for extreme genetic diversity
                "crossover_rate": 1.0,  # Ensuring crossover in every generation for full genetic mixing
                "elite_rate": 0.98,  # Elevating elitism rate to preserve nearly all top solutions
                "diversification_rate": 0.7,  # Increasing diversification rate for optimal escape from local optima
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

    def diversify(self, individual, evals):
        diversification_rate = self.params.get('diversification_rate', 0.7)
        if np.random.rand() < diversification_rate:
            mutation_strength = (self.bounds[1] - self.bounds[0]) * np.random.rand() * 0.1
            individual += np.random.normal(0, mutation_strength, self.dim)
            individual = np.clip(individual, self.bounds[0], self.bounds[1])
            evals += 1
        return individual, evals

    def run(self):
        pop_size = self.params.get('pop_size')
        elite_rate = self.params.get('elite_rate')
        crossover_rate = self.params.get('crossover_rate')
        mutation_rate = self.params.get('mutation_rate')

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        pop_cf = np.array([self.func(ind) for ind in pop])
        evals = pop_size
        elite_size = int(pop_size * elite_rate)

        while evals < self.max_evals:
            sorted_indices = np.argsort(pop_cf)
            pop = pop[sorted_indices]
            pop_cf = pop_cf[sorted_indices]

            new_pop = np.empty_like(pop)
            new_pop_cf = np.empty_like(pop_cf)
            new_pop[:elite_size] = pop[:elite_size]
            new_pop_cf[:elite_size] = pop_cf[:elite_size]

            for i in range(elite_size, pop_size):
                if np.random.rand() < crossover_rate:
                    j = np.random.randint(pop_size)
                    while j == i:
                        j = np.random.randint(pop_size)
                    crossover_point = np.random.randint(0, self.dim)
                    new_pop[i, :crossover_point] = pop[j, :crossover_point]

                if np.random.rand() < mutation_rate:
                    mutation_point = np.random.randint(self.dim)
                    new_pop[i, mutation_point] += np.random.normal(0, 1)
                    evals += 1  # Mutation evaluation

                new_pop[i], evals = self.diversify(new_pop[i], evals)
                new_pop[i] = np.clip(new_pop[i], self.bounds[0], self.bounds[1])
                new_pop_cf[i] = self.func(new_pop[i])
                evals += 1  # Diversification evaluation

                if evals >= self.max_evals:
                    break

            pop = np.copy(new_pop)
            pop_cf = np.copy(new_pop_cf)

        best_index = np.argmin(pop_cf)
        best = pop[best_index]
        best_cf = pop_cf[best_index]

        return best, best_cf

