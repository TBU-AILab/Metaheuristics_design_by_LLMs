#To further improve the algorithm, we should focus on the following aspects:
#
#1. **Enhanced Adaptation Mechanisms**: Refine the adaptation mechanisms to ensure better control over particle dynamics.
#2. **Hybrid Local Search**: Integrate a more sophisticated hybrid local search, blending multiple strategies.
#3. **Dynamic Parameter Adjustment**: Implement more refined dynamic parameter adjustment based on performance trends.
#4. **Run Time Analysis and Parameter Tweaking**: Analyze the runtime performance and tweak parameters more dynamically.
#
#Here’s an improved version of the algorithm with these enhancements:
#
import random
import math

def run(func, dim, bounds, max_evals):
    class Particle:
        def __init__(self, position, velocity):
            self.position = position
            self.velocity = velocity
            self.best_pos = list(position)
            self.best_cost = func(position)
            self.current_cost = self.best_cost

    def initialize(num_particles):
        particles = []
        for _ in range(num_particles):
            position = [random.uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
            velocity = [random.uniform(-0.1, 0.1) for _ in range(dim)]
            particles.append(Particle(position, velocity))
        return particles

    def update_velocity(particle, global_best_pos, w, c1, c2):
        new_velocity = []
        for d in range(dim):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (particle.best_pos[d] - particle.position[d])
            social = c2 * r2 * (global_best_pos[d] - particle.position[d])
            new_v = w * particle.velocity[d] + cognitive + social
            
            # Ensure velocity is within bounds
            position_next = particle.position[d] + new_v
            if position_next < bounds[d][0] or position_next > bounds[d][1]:
                new_v = random.uniform(-abs(bounds[d][1] - bounds[d][0]), abs(bounds[d][1] - bounds[d][0]))
              
            new_velocity.append(new_v)
        return new_velocity

    def update_position(particle):
        new_position = []
        for d in range(dim):
            new_pos = particle.position[d] + particle.velocity[d]
            if new_pos < bounds[d][0]:
                new_pos = bounds[d][0]
            elif new_pos > bounds[d][1]:
                new_pos = bounds[d][1]
            new_position.append(new_pos)
        return new_position

    def hybrid_local_search(current_pos, best_pos):
        # Small perturbation (exploitation)
        perturbation1 = [random.uniform(-0.01, 0.01) * (bounds[d][1] - bounds[d][0]) for d in range(dim)]
        new_position1 = [current_pos[d] + perturbation1[d] for d in range(dim)]
        
        # Larger perturbation (exploration)
        perturbation2 = [random.uniform(-0.1, 0.1) * (bounds[d][1] - bounds[d][0]) for d in range(dim)]
        new_position2 = [best_pos[d] + perturbation2[d] for d in range(dim)]
        
        # Ensuring new positions are within bounds
        for d in range(dim):
            new_position1[d] = max(min(new_position1[d], bounds[d][1]), bounds[d][0])
            new_position2[d] = max(min(new_position2[d], bounds[d][1]), bounds[d][0])

        cost1 = func(new_position1)
        cost2 = func(new_position2)
        
        if cost1 < cost2 and cost1 < func(best_pos):
            return new_position1
        elif cost2 < func(best_pos):
            return new_position2
        else:
            return best_pos

    def adaptive_parameter_control(global_best_cost, previous_global_best_cost, w, c1, c2):
        if global_best_cost < previous_global_best_cost:
            w *= 0.98
            c1 *= 1.02
            c2 *= 1.02
        else:
            w *= 1.02
            c1 *= 0.98
            c2 *= 0.98
        
        w = min(max(w, 0.3), 0.9)
        c1 = min(max(c1, 1.5), 2.5)
        c2 = min(max(c2, 1.5), 2.5)
        
        return w, c1, c2

    def detect_stagnation(improvement_history, threshold=20):
        if len(improvement_history) < threshold:
            return False
        return all(improvement == 0 for improvement in improvement_history[-threshold:])

    def intensive_local_search(elite_particles):
        for particle in elite_particles:
            current_pos = particle.position
            new_pos = hybrid_local_search(current_pos, particle.best_pos)
            new_cost = func(new_pos)

            if new_cost < particle.best_cost:
                particle.position = new_pos
                particle.best_cost = new_cost
                particle.best_pos = new_pos
                
    num_particles = 40
    w_init, w_min = 0.9, 0.3
    c1, c2 = 2.0, 2.0
    max_no_improvement = 50

    particles = initialize(num_particles)
    global_best_pos = particles[0].best_pos
    global_best_cost = particles[0].best_cost
    num_evals = len(particles)
    previous_global_best_cost = global_best_cost
    improvement_history = []
    no_improvement_count = 0

    while num_evals < max_evals and no_improvement_count < max_no_improvement:
        for particle in particles:
            if num_evals >= max_evals:
                break
                
            w = w_min + (w_init - w_min) * (1 - num_evals / max_evals) ** 2
            particle.velocity = update_velocity(particle, global_best_pos, w, c1, c2)
            particle.position = update_position(particle)
            particle.current_cost = func(particle.position)
            num_evals += 1

            if particle.current_cost < particle.best_cost:
                particle.best_cost = particle.current_cost
                particle.best_pos = list(particle.position)

                if particle.current_cost < global_best_cost:
                    global_best_cost = particle.current_cost
                    global_best_pos = list(particle.position)
                    improvement_history.append(1)
                    no_improvement_count = 0
                else:
                    improvement_history.append(0)
                    no_improvement_count += 1
            else:
                improvement_history.append(0)
                no_improvement_count += 1

        if detect_stagnation(improvement_history, 20) and num_evals < max_evals:
            to_reinitialize = int(len(particles) * 0.2)
            particles.extend(initialize(to_reinitialize))
            particles = particles[:num_particles]
            no_improvement_count = 0

        elite_fraction = 0.2
        elite_particles = sorted(particles, key=lambda p: p.current_cost)[:int(len(particles) * elite_fraction)]
        intensive_local_search(elite_particles)

        w, c1, c2 = adaptive_parameter_control(global_best_cost, previous_global_best_cost, w, c1, c2)
        previous_global_best_cost = global_best_cost

    return global_best_cost
#
#### Summary of Key Enhancements:
#1. **Hybrid Local Search**: Integrates both small and large perturbations to balance exploration and exploitation.
#2. **Adaptive Parameter Control**: Refinement of dynamic parameter adjustments based on performance trends.
#3. **Intensive Local Search for Elites**: An enhanced local search mechanism specific to elite particles.
#4. **Strategic Initialization and Managing Population Size**: Ensures a diverse and robust search space is maintained.
#
#These enhancements aim for a sophisticated balance between exploration and exploitation, more robust parameter controls, and a more strategic approach to local search, potentially yielding better optimization results.
