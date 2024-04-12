import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                "pop_size": 200,  # Increased population size for broader search
                "step_size": 0.03,  # Further reduced step size for precision
                "prt": 0.6,  # Increased perturbation for more exploration
                "path_length": 4.0,  # Increased path length for deeper search
                "mutation_rate": 0.12,  # Adjusted mutation rate for diversity
                "crossover_rate": 0.98,  # High crossover rate for genetic diversity
                "elite_rate": 0.2,  # Increased elitism rate to preserve top performers
                "local_search_rate": 0.05,  # Introduced local search rate
                "local_search_step": 0.01,  # Step size for local search
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

    def local_search(self, individual):
        step = self.params.get('local_search_step', 0.01)
        for d in range(self.dim):
            # Perturb each dimension slightly and check if it improves the fitness
            temp = np.copy(individual)
            temp[d] += step
            temp = np.clip(temp, self.bounds[0], self.bounds[1])
            if self.func(temp) < self.func(individual):
                individual = temp
            else:
                temp[d] -= 2 * step
                temp = np.clip(temp, self.bounds[0], self.bounds[1])
                if self.func(temp) < self.func(individual):
                    individual = temp
        return individual

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

                new_pop[i] = np.clip(new_pop[i], self.bounds[0], self.bounds[1])
                new_pop_cf[i] = self.func(new_pop[i])
                evals += 1

                if np.random.rand() < local_search_rate:
                    new_pop[i] = self.local_search(new_pop[i])
                    new_pop_cf[i] = self.func(new_pop[i])
                    evals += self.dim * 2  # Account for local search evaluations

                if evals >= self.max_evals:
                    break

            pop = np.copy(new_pop)
            pop_cf = np.copy(new_pop_cf)

        best_index = np.argmin(pop_cf)
        best = pop[best_index]
        best_cf = pop_cf[best_index]

        return best, best_cf

