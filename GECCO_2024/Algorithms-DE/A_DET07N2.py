import numpy as np

class Algorithm():

    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100, 'F': 0.8, 'CR': 0.9, 'p': 0.15, 'beta': 0.5,
                'archive_size': 50, 'mutation_strategy': 'rand/2/bin'
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
        fitness = np.asarray([self.func(ind) for ind in solutions])
        self.evals += len(solutions)
        return fitness

    def mutation(self, pop, best, idxs, F):
        if self.params['mutation_strategy'] == 'rand/2/bin':
            a, b, c, d = pop[np.random.choice(idxs, 4, replace=False)]
            mutant = a + F * (b - c) + F * (best - d)
        else:  # Default to 'rand/1/bin'
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
        return mutant

    def run(self):
        pop_size = self.params.get('pop_size', 50)
        F = self.params.get('F', 0.8)
        Cr = self.params.get('CR', 0.9)
        p = self.params.get('p', 0.15)
        beta = self.params.get('beta', 0.5)
        archive_size = self.params.get('archive_size', 50)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)

        best_index = np.argmin(fitness)
        best = pop[best_index]

        while self.evals < self.max_evals:
            new_archive = []
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                mutant = np.clip(self.mutation(pop, best, idxs, F), self.bounds[0], self.bounds[1])

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

                    if trial_fitness < fitness[best_index]:
                        best_index = i
                        best = trial

                if self.evals >= self.max_evals:
                    break

            # Update archive
            self.archive = (self.archive + new_archive)[-archive_size:]

            # Adaptive parameter adjustment
            F = F * (1 - beta) + beta * np.random.rand()
            Cr = Cr * (1 - beta) + beta * np.random.rand()

        return best, fitness[best_index]

