import numpy as np

class Algorithm():

    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100, 'F': 0.8, 'CR': 0.9, 'p': 0.11, 'beta': 0.4,
                'archive_size': 100, 'mutation_strategy': 'current-to-pbest/1/bin',
                'adaptive': True, 'local_search_prob': 0.01, 'local_search_step': 0.1,
                'elite_size': 5, 'elite_local_search_prob': 0.05, 'memory': True,
                'memory_size': 5, 'diversification_prob': 0.05, 'landscape_analysis_interval': 50
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.archive = []
        self.memory = []
        self.generation = 0

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
        pbest_idx = np.random.choice(idxs, max(1, int(len(idxs) * self.params['p'])), replace=False)
        pbest_fitness = [self.func(pop[i]) for i in pbest_idx[:self.max_evals - self.evals]]
        self.evals += len(pbest_fitness)
        if self.evals >= self.max_evals:
            return None
        pbest = pop[pbest_idx[np.argmin(pbest_fitness)]]
        a, b = pop[np.random.choice(idxs, 2, replace=False)]
        mutant = pop[idxs[0]] + F * (pbest - pop[idxs[0]]) + F * (a - b)
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

    def landscape_analysis(self, pop):
        diffs = []
        for _ in range(min(100, len(pop))):
            i, j = np.random.choice(len(pop), 2, replace=False)
            diff = abs(self.func(pop[i]) - self.func(pop[j]))
            diffs.append(diff)
            self.evals += 2
            if self.evals >= self.max_evals:
                break
        ruggedness = np.var(diffs)
        return ruggedness

    def adapt_parameters(self, ruggedness):
        if ruggedness > 0.1:
            self.params['F'] *= 1.1
            self.params['CR'] *= 0.9
        else:
            self.params['F'] *= 0.9
            self.params['CR'] *= 1.1
        self.params['F'] = np.clip(self.params['F'], 0.1, 1)
        self.params['CR'] = np.clip(self.params['CR'], 0.1, 1)

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        F = self.params.get('F', 0.8)
        Cr = self.params.get('CR', 0.9)
        elite_size = self.params.get('elite_size', 5)
        p = self.params.get('p', 0.11)

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)

        while self.evals < self.max_evals:
            new_archive = []
            elite_indices = np.argpartition(fitness, elite_size)[:elite_size]

            for i in range(pop_size):
                if self.evals >= self.max_evals:
                    break
                idxs = [idx for idx in range(pop_size) if idx != i]
                mutant = self.mutation(pop, pop[elite_indices[0]], idxs, F, p)
                if mutant is None:
                    break

                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = self.func(trial)
                self.evals += 1
                if trial_fitness < fitness[i]:
                    new_archive.append(pop[i])
                    pop[i] = trial
                    fitness[i] = trial_fitness

            if self.generation % self.params['landscape_analysis_interval'] == 0:
                ruggedness = self.landscape_analysis(pop)
                self.adapt_parameters(ruggedness)

            self.generation += 1

        best_index = np.argmin(fitness)
        return pop[best_index], fitness[best_index]

