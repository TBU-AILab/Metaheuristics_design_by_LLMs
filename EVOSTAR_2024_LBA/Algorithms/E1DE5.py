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
            'diversification_phase': True,  # Enable a diversification phase
            'intensification_phase': True,  # Enable an intensification phase
            'diversification_trigger': 0.25,  # Trigger for diversification
            'intensification_trigger': 0.75,  # Trigger for intensification
        }

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F_base = self.params.get('F', 0.5)
        Cr_base = self.params.get('CR', 0.7)
        adaptive = self.params.get('adaptive', False)
        diversification_phase = self.params.get('diversification_phase', True)
        intensification_phase = self.params.get('intensification_phase', True)
        diversification_trigger = self.params.get('diversification_trigger', 0.25)
        intensification_trigger = self.params.get('intensification_trigger', 0.75)

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
            phase_progress = evals / self.max_evals

            if adaptive:
                # Dynamically adjust F and Cr based on phase progress
                F_base = np.clip(F_base + 0.01 * np.sin(phase_progress * np.pi * 2), 0.1, 1)
                Cr_base = np.clip(Cr_base + 0.01 * np.cos(phase_progress * np.pi * 2), 0, 1)

            if diversification_phase and phase_progress < diversification_trigger:
                # Increase population diversity
                mutation_scale = F_base * (1 + 0.5 * np.sin(phase_progress * np.pi))
            elif intensification_phase and phase_progress > intensification_trigger:
                # Focus on intensifying search around current best
                mutation_scale = F_base * (1 - 0.5 * np.cos(phase_progress * np.pi))
            else:
                mutation_scale = F_base

            for i in range(pop_size):
                # Mutation and Crossover
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + mutation_scale * (b - c), self.bounds[0], self.bounds[1])
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

