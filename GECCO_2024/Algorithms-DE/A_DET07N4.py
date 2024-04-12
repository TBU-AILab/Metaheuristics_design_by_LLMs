import numpy as np

class Algorithm():

    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100, 'F': 0.8, 'CR': 0.9, 'p': 0.11, 'beta': 0.4,
                'archive_size': 100, 'mutation_strategy': 'current-to-pbest/1/bin',
                'adaptive': True, 'local_search_prob': 0.01, 'local_search_step': 0.1
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.archive = []

    def evaluate(self, solutions):
        if self.evals + len(solutions) > self.max_evals:
            solutions = solutions[:self.max_evals - self.evals]
        fitness = np.asarray([self.func(ind) for ind in solutions])
        self.evals += len(solutions)
        return fitness

    def mutation(self, pop, best, idxs, F, p):
        if self.params['mutation_strategy'] == 'current-to-pbest/1/bin':
            pbest_idx = np.random.choice(idxs, max(1, int(len(idxs) * p)), replace=False)
            pbest_fitness = [self.func(pop[i]) for i in pbest_idx[:self.max_evals - self.evals]]
            self.evals += len(pbest_fitness)  # Increment evals for each function evaluation
            pbest = pop[pbest_idx[np.argmin(pbest_fitness)]]
            a, b = pop[np.random.choice(idxs, 2, replace=False)]
            mutant = pop[idxs[0]] + F * (pbest - pop[idxs[0]]) + F * (a - b)
        elif self.params['mutation_strategy'] == 'rand/2/bin':
            a, b, c, d = pop[np.random.choice(idxs, 4, replace=False)]
            mutant = a + F * (b - c) + F * (best - d)
        else:  # Default to 'rand/1/bin'
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
        return mutant

    def local_search(self, best, step_size):
        candidate = best + np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
        candidate_fitness = self.func(candidate)
        self.evals += 1
        return candidate, candidate_fitness

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        F = self.params.get('F', 0.8)
        Cr = self.params.get('CR', 0.9)
        p = self.params.get('p', 0.11)
        beta = self.params.get('beta', 0.4)
        archive_size = self.params.get('archive_size', 100)
        adaptive = self.params.get('adaptive', True)
        local_search_prob = self.params.get('local_search_prob', 0.01)
        local_search_step = self.params.get('local_search_step', 0.1)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)

        best_index = np.argmin(fitness)
        best = pop[best_index]

        while self.evals < self.max_evals:
            new_archive = []
            for i in range(pop_size):
                if self.evals >= self.max_evals:
                    return best, fitness[best_index]
                idxs = [idx for idx in range(pop_size) if idx != i]
                mutant = np.clip(self.mutation(pop, best, idxs, F, p), self.bounds[0], self.bounds[1])

                # Crossover: bin
                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                if self.evals < self.max_evals:
                    trial_fitness = self.func(trial)
                    self.evals += 1
                    if trial_fitness < fitness[i]:
                        new_archive.append(pop[i])
                        pop[i] = trial
                        fitness[i] = trial_fitness

                        if trial_fitness < fitness[best_index]:
                            best_index = i
                            best = trial

                # Local Search
                if np.random.rand() < local_search_prob and self.evals < self.max_evals:
                    local_candidate, local_fitness = self.local_search(best, local_search_step)
                    if local_fitness < fitness[best_index]:
                        best = local_candidate
                        fitness[best_index] = local_fitness

            # Update archive
            self.archive = (self.archive + new_archive)[-archive_size:]

            if adaptive:
                # Adaptive parameter adjustment
                F = np.clip(F * (1 - beta) + beta * np.random.rand(), 0.1, 1)
                Cr = np.clip(Cr * (1 - beta) + beta * np.random.rand(), 0.1, 1)

        return best, fitness[best_index]

