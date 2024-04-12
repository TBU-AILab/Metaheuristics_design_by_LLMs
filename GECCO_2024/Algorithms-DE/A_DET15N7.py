import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 200,  # Population size maintained
                'F_base': 0.5,  # Base value for F
                'F_amp': 0.3,  # Amplitude for F variation
                'CR_base': 0.9,  # Base value for CR
                'CR_amp': 0.05,  # Amplitude for CR variation
                'p': 0.25,  # Probability for choosing best individuals
                'archive_size': 300,  # Increased size of the archive
                'mutation_strategy': 'best/2/bin',  # Mutation strategy maintained
                'adaptive': True,  # Adaptive parameters enabled
                'F_adapt': True,  # Adaptive F enabled
                'CR_adapt': True,  # Adaptive CR enabled
                'memory': True,  # Memory for F and CR enabled
                'elite_strategy': True,  # Elite strategy enabled
                'elite_size': 5,  # Size of elite members
                'diversification': True,  # Diversification strategy enabled
                'diversification_rate': 0.1,  # Rate of diversification
                'local_search': True,  # Introducing local search
                'local_search_rate': 0.05,  # Rate of local search
                'local_search_step': 0.1,  # Step size for local search
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

        # Memory for adaptive parameters
        self.F_memory = [0.5] * 5
        self.CR_memory = [0.9] * 5

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F_base = self.params.get('F_base', 0.5)
        F_amp = self.params.get('F_amp', 0.3)
        CR_base = self.params.get('CR_base', 0.9)
        CR_amp = self.params.get('CR_amp', 0.05)
        p = self.params.get('p', 0.1)
        archive_size = self.params.get('archive_size', pop_size)
        mutation_strategy = self.params.get('mutation_strategy', 'rand/1')
        adaptive = self.params.get('adaptive', False)
        F_adapt = self.params.get('F_adapt', False)
        CR_adapt = self.params.get('CR_adapt', False)
        memory = self.params.get('memory', False)
        elite_strategy = self.params.get('elite_strategy', False)
        elite_size = self.params.get('elite_size', 5)
        diversification = self.params.get('diversification', False)
        diversification_rate = self.params.get('diversification_rate', 0.1)
        local_search = self.params.get('local_search', False)
        local_search_rate = self.params.get('local_search_rate', 0.05)
        local_search_step = self.params.get('local_search_step', 0.1)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size
        archive = []

        best_index = np.argmin(fitness)
        best = fitness[best_index]

        while evals < self.max_evals:
            if diversification and np.random.rand() < diversification_rate:
                # Apply diversification
                diversification_indices = np.random.choice(range(pop_size), size=int(pop_size * diversification_rate), replace=False)
                for idx in diversification_indices:
                    pop[idx] = np.random.rand(self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
                    fitness[idx] = self.func(pop[idx])
                    evals += 1
                    if fitness[idx] < best:
                        best = fitness[idx]

            for i in range(pop_size):
                if adaptive:
                    # Adaptive F and Cr with memory
                    if memory:
                        F = np.clip(np.random.normal(np.mean(self.F_memory), F_amp), 0, 1) if F_adapt else F_base
                        Cr = np.clip(np.random.normal(np.mean(self.CR_memory), CR_amp), 0, 1) if CR_adapt else CR_base
                    else:
                        F = np.clip(np.random.normal(F_base, F_amp), 0, 1) if F_adapt else F_base
                        Cr = np.clip(np.random.normal(CR_base, CR_amp), 0, 1) if CR_adapt else CR_base

                # Mutation
                if mutation_strategy == 'best/2/bin':
                    best_idx = np.argmin(fitness)
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b, c, d = pop[np.random.choice(idxs, 4, replace=False)]
                    mutant = np.clip(pop[best_idx] + F * (a + b - c - d), self.bounds[0], self.bounds[1])
                else:  # Default mutation strategy (rand/1)
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(np.random.randint(0, len(archive)))
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best:
                        best = trial_fitness
                        # Update memory
                        if memory:
                            self.F_memory.pop(0)
                            self.F_memory.append(F)
                            self.CR_memory.pop(0)
                            self.CR_memory.append(Cr)

                if evals >= self.max_evals:
                    break

            # Local search
            if local_search and np.random.rand() < local_search_rate:
                local_search_indices = np.random.choice(range(pop_size), size=int(pop_size * local_search_rate), replace=False)
                for idx in local_search_indices:
                    local_point = pop[idx] + np.random.uniform(-local_search_step, local_search_step, self.dim)
                    local_point = np.clip(local_point, self.bounds[0], self.bounds[1])
                    local_fitness = self.func(local_point)
                    evals += 1
                    if local_fitness < fitness[idx]:
                        pop[idx] = local_point
                        fitness[idx] = local_fitness
                        if local_fitness < best:
                            best = local_fitness

        return best

