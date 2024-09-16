#Here's an improved version of the algorithm:
#
import numpy as np

def run(func, dim, bounds, max_evals):
    population_size = min(100, max_evals // 4)
    best = float('inf')
    
    lower_bounds = np.array([bound[0] for bound in bounds])
    upper_bounds = np.array([bound[1] for bound in bounds])
    
    population = np.random.uniform(lower_bounds, upper_bounds, (population_size, dim))
    velocities = np.zeros_like(population)
    fitness = np.array([func(ind) for ind in population])
    
    personal_best = population.copy()
    personal_best_fitness = fitness.copy()
    
    global_best = population[np.argmin(fitness)]
    global_best_fitness = np.min(fitness)
    
    w = 0.9  # inertia weight
    c1 = 2.05  # cognitive parameter
    c2 = 2.05  # social parameter
    phi = c1 + c2
    chi = 2 / (phi - 2 + np.sqrt(phi**2 - 4*phi))  # constriction factor
    
    F = 0.5  # differential weight
    CR = 0.9  # crossover rate
    
    evals = population_size
    stagnation_counter = 0
    
    while evals < max_evals:
        for i in range(population_size):
            if evals >= max_evals:
                break
            
            r1, r2 = np.random.rand(2)
            velocities[i] = chi * (velocities[i] +
                                   c1 * r1 * (personal_best[i] - population[i]) +
                                   c2 * r2 * (global_best - population[i]))
            
            a, b, c = np.random.choice(population_size, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            
            cross_points = np.random.rand(dim) < CR
            trial = np.where(cross_points, mutant, population[i] + velocities[i])
            
            trial = np.clip(trial, lower_bounds, upper_bounds)
            
            trial_fitness = func(trial)
            evals += 1
            
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness
                    
                    if trial_fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = trial_fitness
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
            else:
                stagnation_counter += 1
        
        best = min(best, global_best_fitness)
        
        # Adaptive parameter update
        F = np.clip(np.random.normal(0.5, 0.3), 0.1, 1.0)
        CR = np.clip(np.random.normal(0.9, 0.1), 0.5, 1.0)
        
        # Local search for the best solution
        if stagnation_counter > 10:
            local_search_point = global_best + np.random.normal(0, 0.1, dim)
            local_search_point = np.clip(local_search_point, lower_bounds, upper_bounds)
            local_search_fitness = func(local_search_point)
            evals += 1
            
            if local_search_fitness < global_best_fitness:
                global_best = local_search_point
                global_best_fitness = local_search_fitness
                stagnation_counter = 0
        
        # Reinitialize worst solution if stagnation persists
        if stagnation_counter > 20:
            worst_idx = np.argmax(fitness)
            population[worst_idx] = np.random.uniform(lower_bounds, upper_bounds)
            fitness[worst_idx] = func(population[worst_idx])
            evals += 1
            stagnation_counter = 0
    
    return best
