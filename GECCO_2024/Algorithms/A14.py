import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                "pop_size": 300,  # Adjusted population size for broader initial search
                "step_size": 0.01,  # Reduced step size for even finer exploration
                "prt": 0.75,  # Increased perturbation for greater exploration
                "path_length": 5.0,  # Increased path length for extended search
                "mutation_rate": 0.2,  # Slightly increased mutation rate
                "crossover_rate": 1.0,  # Ensuring crossover happens every time
                "elite_rate": 0.3,  # Increased elitism rate to preserve more top performers
                "local_search_rate": 0.2,  # Increased local search rate for more frequent refinement
                "local_search_step": 0.002,  # Reduced step size for local search for precision
                "diversification_rate": 0.05,  # Increased diversification rate to escape local optima
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

    def local_search(self, individual, evals):
        step = self.params.get('local_search_step', 0.01)
        for d in range(self.dim):
            temp = np.copy(individual)
            temp[d] += step
            temp = np.clip(temp, self.bounds[0], self.bounds[1])
            evals += 1
            temp_cf = self.func(temp)
            if temp_cf < self.func(individual):
                individual = temp
            else:
                temp[d] -= 2 * step
                temp = np.clip(temp, self.bounds[0], self.bounds[1])
                evals += 1
                temp_cf = self.func(temp)
                if temp_cf < self.func(individual):
                    individual = temp
        return individual, evals

    def diversify(self, individual, evals):
        diversification_rate = self.params.get('diversification_rate', 0.02)
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
        local_search_rate = self.params.get('local_search_rate')

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
                    evals += 1

                new_pop[i], evals = self.diversify(new_pop[i], evals)
                new_pop[i] = np.clip(new_pop[i], self.bounds[0], self.bounds[1])
                new_pop_cf[i] = self.func(new_pop[i])
                evals += 1

                if np.random.rand() < local_search_rate:
                    new_pop[i], evals = self.local_search(new_pop[i], evals)
                    new_pop_cf[i] = self.func(new_pop[i])
                    evals += 1

                if evals >= self.max_evals:
                    break

            pop = np.copy(new_pop)
            pop_cf = np.copy(new_pop_cf)

        best_index = np.argmin(pop_cf)
        best = pop[best_index]
        best_cf = pop_cf[best_index]

        return best, best_cf

