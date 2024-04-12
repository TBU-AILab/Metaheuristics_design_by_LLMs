import numpy as np

class Algorithm():

    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {'pop_size': 50, 'F': 0.5, 'CR': 0.9, 'p': 0.1, 'beta': 0.5}
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
        pop_size = self.params.get('pop_size', 30)
        F = self.params.get('F', 0.5)
        Cr = self.params.get('CR', 0.9)
        p = self.params.get('p', 0.1)
        beta = self.params.get('beta', 0.5)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)

        best_index = np.argmin(fitness)
        best = pop[best_index]

        while self.evals < self.max_evals:
            for i in range(pop_size):
                # Mutation: DE/current-to-pbest/1
                idxs = [idx for idx in range(pop_size) if idx != i]
                pbest = np.random.choice(idxs, int(pop_size * p) + 1)
                pbest = pop[pbest[np.argmin(fitness[pbest])]]
                a, b = pop[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(pbest + F * (a - b), self.bounds[0], self.bounds[1])

                # Crossover: bin
                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                self.evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < fitness[best_index]:
                        best_index = i
                        best = trial

                if self.evals >= self.max_evals:
                    break

            # Adaptive parameter adjustment
            F = F * (1 - beta) + beta * np.random.rand()
            Cr = Cr * (1 - beta) + beta * np.random.rand()

        return best, fitness[best_index]

