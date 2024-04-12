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
                'elite_size': 5,
                'elite_influence': 0.1,
                'hybridization_factor': 0.5,
                'hybridization_probability': 0.3,
                'FLA_step': 0.1,
                'FLA_absorption': 1.0,
                'PSO_weight': 0.5,  # New parameter for PSO inertia weight
                'PSO_c1': 2.0,  # New parameter for PSO cognitive component
                'PSO_c2': 2.0  # New parameter for PSO social component
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
        self.velocity = np.zeros((self.params['pop_size'], self.dim))  # Initialize velocity for PSO

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

    def hybridization(self, individual, best, factor):
        return individual + factor * (best - individual)

    def firefly_movement(self, firefly_i, firefly_j, attractiveness):
        step = self.params['FLA_step'] * (np.random.rand(self.dim) - 0.5) * (self.bounds[1] - self.bounds[0])
        movement = attractiveness * (firefly_j - firefly_i) + step
        return np.clip(firefly_i + movement, self.bounds[0], self.bounds[1])

    def pso_update(self, individual, personal_best, global_best, velocity, i):
        r1, r2 = np.random.rand(), np.random.rand()
        new_velocity = self.params['PSO_weight'] * velocity[i] + \
                       self.params['PSO_c1'] * r1 * (personal_best - individual) + \
                       self.params['PSO_c2'] * r2 * (global_best - individual)
        new_position = np.clip(individual + new_velocity, self.bounds[0], self.bounds[1])
        self.velocity[i] = new_velocity
        return new_position

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        alpha = self.params.get('alpha', 0.5)
        p = self.params.get('p', 0.2)
        gamma = self.params.get('gamma', 0.05)
        beta = self.params.get('beta', 0.001)

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = self.evaluate(pop)
        personal_best = pop.copy()
        personal_best_fitness = fitness.copy()
        
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

                if np.random.rand() < self.params['hybridization_probability']:
                    trial = self.hybridization(trial, best, self.params['hybridization_factor'])

                # PSO update for further exploration and exploitation
                trial = self.pso_update(trial, personal_best[i], best, self.velocity, i)

                trial_fitness = self.func(trial)
                self.evals += 1

                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    self.update_memory_F(F)
                    self.update_CR_memory(CR)
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
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

