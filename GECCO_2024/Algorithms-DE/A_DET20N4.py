import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100,
                'F_base': 0.5,
                'CR': 0.9,
                'alpha': 0.5,
                'p': 0.2,
                'gamma': 0.05,
                'beta': 0.001,
                'memory_size': 5,
                'epsilon': 1e-10  # Parameter to control exploration
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.memory_F = np.full(self.params['memory_size'], self.params['F_base'])
        self.best_fitness_history = []

    def evaluate(self, solutions):
        fitness = np.asarray([self.func(ind) for ind in solutions])
        self.evals += len(solutions)
        return fitness

    def adaptive_F(self):
        return np.random.choice(self.memory_F)

    def update_memory_F(self, F):
        self.memory_F = np.roll(self.memory_F, -1)
        self.memory_F[-1] = F

    def exploration_control(self, gen):
        # Dynamically adjust exploration parameter epsilon based on generation
        return self.params['epsilon'] * (1 - gen / (self.max_evals / self.params['pop_size']))

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        CR = self.params.get('CR', 0.9)
        alpha = self.params.get('alpha', 0.5)
        p = self.params.get('p', 0.2)
        gamma = self.params.get('gamma', 0.05)
        beta = self.params.get('beta', 0.001)

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)
        
        best_index = np.argmin(fitness)
        best = pop[best_index]
        best_fitness = fitness[best_index]
        self.best_fitness_history.append(best_fitness)

        while self.evals < self.max_evals:
            epsilon = self.exploration_control(self.evals)
            for i in range(pop_size):
                F = self.adaptive_F()
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) <= CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                if np.random.rand() < p:
                    local_search_vector = best + alpha * (np.random.rand(self.dim) - 0.5) * (self.bounds[1] - self.bounds[0])
                    trial = np.clip(local_search_vector, self.bounds[0], self.bounds[1])

                trial_fitness = self.func(trial)
                self.evals += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    self.update_memory_F(F)
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness
                        self.best_fitness_history.append(best_fitness)

                if np.random.rand() < gamma or np.abs(best_fitness - fitness[i]) < epsilon:
                    pop[i] = np.random.rand(self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                    fitness[i] = self.func(pop[i])
                    self.evals += 1

                if self.evals % (10 * pop_size) == 0:
                    fitness += beta * np.abs(fitness - best_fitness)

                if self.evals >= self.max_evals:
                    break

        return best_fitness

