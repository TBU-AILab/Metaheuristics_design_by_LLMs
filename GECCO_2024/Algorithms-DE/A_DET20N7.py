import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100,
                'F_base': 0.5,
                'CR_base': 0.9,
                'alpha': 0.5,
                'p': 0.2,
                'gamma': 0.05,
                'beta': 0.001,
                'memory_size': 5,
                'epsilon': 1e-10,
                'learning_period': 50,
                'mutation_strategy': 'current-to-best/2/bin',
                'adaptive_CR': True,
                'CR_memory_size': 5,
                'elite_size': 5,  # New parameter for elite population size
                'elite_influence': 0.1  # New parameter for elite influence in mutation
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.evals = 0
        self.memory_F = np.full(self.params['memory_size'], self.params['F_base'])
        self.CR_memory = np.full(self.params['CR_memory_size'], self.params['CR_base'])
        self.best_fitness_history = []

    def evaluate(self, solutions):
        fitness = np.asarray([self.func(ind) for ind in solutions])
        self.evals += len(solutions)
        return fitness

    def adaptive_F(self):
        return np.random.choice(self.memory_F)

    def adaptive_CR(self):
        return np.random.choice(self.CR_memory)

    def update_memory_F(self, F):
        self.memory_F = np.roll(self.memory_F, -1)
        self.memory_F[-1] = F

    def update_CR_memory(self, CR):
        self.CR_memory = np.roll(self.CR_memory, -1)
        self.CR_memory[-1] = CR

    def exploration_control(self, gen):
        return self.params['epsilon'] * (1 - gen / (self.max_evals / self.params['pop_size']))

    def mutation(self, pop, idx, F, best, elite_pop):
        if self.params['mutation_strategy'] == 'current-to-best/2/bin':
            idxs = [i for i in range(len(pop)) if i != idx]
            a, b, c, d = pop[np.random.choice(idxs, 4, replace=False)]
            elite = elite_pop[np.random.randint(0, len(elite_pop))]
            mutant = pop[idx] + F * (best - pop[idx] + b - c + d - a + self.params['elite_influence'] * (elite - pop[idx]))
        else:
            idxs = [i for i in range(len(pop)) if i != idx]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = a + F * (b - c)
        return mutant

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        alpha = self.params.get('alpha', 0.5)
        p = self.params.get('p', 0.2)
        gamma = self.params.get('gamma', 0.05)
        beta = self.params.get('beta', 0.001)

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)
        
        elite_size = self.params['elite_size']
        elite_indices = np.argpartition(fitness, elite_size)[:elite_size]
        elite_pop = pop[elite_indices]

        best_index = np.argmin(fitness)
        best = pop[best_index]
        best_fitness = fitness[best_index]
        self.best_fitness_history.append(best_fitness)

        while self.evals < self.max_evals:
            epsilon = self.exploration_control(self.evals)
            for i in range(pop_size):
                F = self.adaptive_F()
                CR = self.adaptive_CR() if self.params['adaptive_CR'] else self.params['CR_base']
                mutant = self.mutation(pop, i, F, best, elite_pop)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

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
                    self.update_CR_memory(CR)
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness
                        self.best_fitness_history.append(best_fitness)

                if np.random.rand() < gamma or np.abs(best_fitness - fitness[i]) < epsilon:
                    pop[i] = np.random.rand(self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                    fitness[i] = self.func(pop[i])
                    self.evals += 1

                if self.evals % (self.params['learning_period'] * pop_size) == 0:
                    fitness += beta * np.abs(fitness - best_fitness)

                elite_indices = np.argpartition(fitness, elite_size)[:elite_size]
                elite_pop = pop[elite_indices]

                if self.evals >= self.max_evals:
                    break

        return best_fitness

