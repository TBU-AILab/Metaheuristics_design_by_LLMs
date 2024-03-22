import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.params = {
            'pop_size': 20,
            'F': 0.8,
            'CR': 0.9,
            'adaptive': True,
            'dynamic_topology': True,  # Enable dynamic population topology
            'topology_update_freq': 1000,  # Frequency of topology updates
            'elite_focus': True,  # Focus more on elite individuals
            'elite_ratio': 0.1,  # Ratio of population considered elite
        }

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F_base = self.params.get('F', 0.5)
        Cr_base = self.params.get('CR', 0.7)
        adaptive = self.params.get('adaptive', False)
        dynamic_topology = self.params.get('dynamic_topology', False)
        topology_update_freq = self.params.get('topology_update_freq', 1000)
        elite_focus = self.params.get('elite_focus', False)
        elite_ratio = self.params.get('elite_ratio', 0.1)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size

        best_index = np.argmin(fitness)
        best = {
            'params': pop[best_index],
            'fitness': fitness[best_index],
            'gen': 0,
            'eval_num': evals
        }

        while evals < self.max_evals:
            if adaptive:
                # Dynamically adjust F and Cr based on performance
                F_base = np.clip(F_base + 0.01 * np.sin(evals / 1000), 0.1, 1)
                Cr_base = np.clip(Cr_base + 0.01 * np.cos(evals / 1000), 0, 1)

            if dynamic_topology and evals % topology_update_freq == 0:
                # Update population topology by shuffling
                np.random.shuffle(pop)

            elite_count = int(pop_size * elite_ratio)
            elite_indices = np.argpartition(fitness, elite_count)[:elite_count]

            for i in range(pop_size):
                if elite_focus and i in elite_indices:
                    # For elite individuals, use a more aggressive mutation strategy
                    F = F_base + 0.2
                else:
                    F = F_base

                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) <= Cr_base
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best['fitness']:
                        best['fitness'] = trial_fitness
                        best['params'] = trial
                        best['gen'] = evals // pop_size
                        best['eval_num'] = evals

                if evals >= self.max_evals:
                    break

        return best

