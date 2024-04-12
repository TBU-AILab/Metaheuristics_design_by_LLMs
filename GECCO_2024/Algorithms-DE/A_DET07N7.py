import numpy as np

class Algorithm():

    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100, 'F': 0.8, 'CR': 0.9, 'p': 0.11, 'beta': 0.4,
                'archive_size': 100, 'mutation_strategy': 'current-to-pbest/1/bin',
                'adaptive': True, 'local_search_prob': 0.01, 'local_search_step': 0.1,
                'elite_size': 5, 'elite_local_search_prob': 0.05, 'memory': True,
                'memory_size': 5, 'diversification_prob': 0.05
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.archive = []
        self.memory = []  # Initialize memory for storing best solutions

    def evaluate(self, solutions):
        remaining_evals = self.max_evals - self.evals
        if len(solutions) > remaining_evals:
            solutions = solutions[:remaining_evals]
        fitness = np.asarray([self.func(ind) for ind in solutions])
        self.evals += len(solutions)
        return fitness

    def mutation(self, pop, best, idxs, F, p):
        if self.evals >= self.max_evals:
            return None
        if self.params['mutation_strategy'] == 'current-to-pbest/1/bin':
            pbest_idx = np.random.choice(idxs, max(1, int(len(idxs) * p)), replace=False)
            pbest_fitness = [self.func(pop[i]) for i in pbest_idx[:self.max_evals - self.evals]]
            self.evals += len(pbest_fitness)  # Increment evals for each function evaluation
            if self.evals >= self.max_evals:
                return None
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

    def local_search(self, candidate, step_size):
        if self.evals >= self.max_evals:
            return candidate, np.inf
        candidate = candidate + np.random.uniform(-step_size, step_size, self.dim)
        candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
        candidate_fitness = self.func(candidate)
        self.evals += 1
        return candidate, candidate_fitness

    def update_memory(self, best):
        if len(self.memory) < self.params['memory_size']:
            self.memory.append(best)
        else:
            self.memory.pop(0)
            self.memory.append(best)

    def recall_memory(self):
        if self.memory:
            return np.mean(self.memory, axis=0)
        else:
            return None

    def diversification(self, pop):
        return np.random.rand(len(pop), self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

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
        elite_size = self.params.get('elite_size', 5)
        elite_local_search_prob = self.params.get('elite_local_search_prob', 0.05)
        diversification_prob = self.params.get('diversification_prob', 0.05)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)

        while self.evals < self.max_evals:
            new_archive = []
            elite_indices = np.argpartition(fitness, elite_size)[:elite_size]

            for i in range(pop_size):
                if self.evals >= self.max_evals:
                    break
                if np.random.rand() < diversification_prob:
                    pop[i] = self.diversification([pop[i]])[0]
                    fitness[i] = self.func(pop[i])
                    self.evals += 1
                    continue

                idxs = [idx for idx in range(pop_size) if idx != i]
                mutant = self.mutation(pop, pop[elite_indices[0]], idxs, F, p)
                if mutant is None:
                    break

                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                # Crossover: bin
                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                self.evals += 1
                if trial_fitness < fitness[i]:
                    new_archive.append(pop[i])
                    pop[i] = trial
                    fitness[i] = trial_fitness

            # Elite Local Search
            for elite_idx in elite_indices:
                if np.random.rand() < elite_local_search_prob and self.evals < self.max_evals:
                    local_candidate, local_fitness = self.local_search(pop[elite_idx], local_search_step)
                    if local_fitness < fitness[elite_idx]:
                        pop[elite_idx] = local_candidate
                        fitness[elite_idx] = local_fitness

            # Update archive and memory
            self.archive = (self.archive + new_archive)[-archive_size:]
            best_index = np.argmin(fitness)
            self.update_memory(pop[best_index])

            if adaptive and self.evals < self.max_evals:
                # Adaptive parameter adjustment
                F = np.clip(F * (1 - beta) + beta * np.random.rand(), 0.1, 1)
                Cr = np.clip(Cr * (1 - beta) + beta * np.random.rand(), 0.1, 1)

        # Recall from memory if needed
        if self.params['memory']:
            best_memory = self.recall_memory()
            if best_memory is not None:
                memory_fitness = self.func(best_memory)
                self.evals += 1
                if memory_fitness < fitness[best_index]:
                    return best_memory, memory_fitness

        return pop[best_index], fitness[best_index]

