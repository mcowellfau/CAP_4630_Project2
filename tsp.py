import random
import math

# Define the City class to represent cities with x and y coordinates
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    # Method to calculate distance between two cities
    def distance(self, city):
        x_distance = abs(self.x - city.x)
        y_distance = abs(self.y - city.y)
        return math.sqrt(x_distance**2 + y_distance**2)
# Definition of the GeneticAlgorithm class to handle TSP solving
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
    # Initialize a random population of routes
    def create_initial_population(self, city_list):
        population = []
        for _ in range(self.population_size):
            population.append(random.sample(city_list, len(city_list)))
        return population
    # Calculate fitness of a route (inverse of total distance)
    def calculate_fitness(self, route):
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += from_city.distance(to_city)
        return 1 / total_distance
    # Select two parents from the population using tournament selection
    def selection(self, population):
        selected_parents = random.sample(population, 2)
        parent1 = max(selected_parents, key=self.calculate_fitness)
        selected_parents.remove(parent1)
        parent2 = selected_parents[0]
        return parent1, parent2
    # Perform crossover operation to create a child route
    def crossover(self, parent1, parent2):
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start + 1, len(parent1))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        for city in parent2:
            if city not in child:
                for i in range(len(child)):
                    if child[i] is None:
                        child[i] = city
                        break
        return child
    # Apply mutation to a route with a certain probability
    def mutation(self, route):
        for i in range(len(route)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(route) - 1)
                route[i], route[j] = route[j], route[i]
        return route
    # Evolve the population by selecting parents, performing crossover and mutation
    def evolve_population(self, population):
        new_population = []
        elitism_offset = 1
        best_route = max(population, key=self.calculate_fitness)
        new_population.append(best_route)
        for _ in range(self.population_size - elitism_offset):
            parent1, parent2 = self.selection(population)
            child = self.crossover(parent1, parent2)
            child = self.mutation(child)
            new_population.append(child)
        return new_population
    # Solve TSP using Genetic Algorithm
    def tsp_ga(self, city_list):
        population = self.create_initial_population(city_list)
        for _ in range(self.generations):
            population = self.evolve_population(population)
        best_route = max(population, key=self.calculate_fitness)
        return best_route
# Function to get user input with an optional default value
def get_user_input(prompt, default=None):
    user_input = input(prompt)
    return default if user_input == '' else user_input
# Defined the main function to orchestrate the solving process
def main():
    # Prompt user for parameters
    num_cities = int(get_user_input("Enter the integer number of cities in the Traveling Salesman Problem (default is 25): ", 25))
    population_size = int(get_user_input("Enter the integer population size: "))
    mutation_rate = float(get_user_input("Enter the float mutation rate: "))
    new_children_proportion = float(get_user_input("Enter the proportion of new children in each generation: "))
    use_stagnation = get_user_input("Would you like to use stagnation as the stopping condition? (Y/N): ").lower() == 'y'
    
    if use_stagnation:
        stagnation_generations = int(get_user_input("Enter the number of consecutive generations with no improvement to use as stopping condition: "))
        num_generations = 500 # Default value when using stagnation
    else:
        num_generations = int(get_user_input("Enter the integer number of generations: "))

    random.seed()  
    # Generate random cities
    city_list = [City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(num_cities)]

    ga = GeneticAlgorithm(population_size=population_size, mutation_rate=mutation_rate, generations=num_generations)

    best_route = None

    if use_stagnation:
        prev_best_fitness = 0
        stagnation_count = 0

        for epoch in range(num_generations):
            population = ga.create_initial_population(city_list)
            best_route = max(population, key=ga.calculate_fitness)
            best_fitness = ga.calculate_fitness(best_route)

            if best_fitness > prev_best_fitness:
                prev_best_fitness = best_fitness
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= stagnation_generations:
                break

            population = ga.evolve_population(population)

            min_distance = min(ga.calculate_fitness(route) for route in population)
            print(f"Epoch {epoch+1}: Minimum Distance = {1/min_distance:.2f} units")

    else:
        population = ga.create_initial_population(city_list)
        for epoch in range(num_generations):
            population = ga.evolve_population(population)
            best_route = max(population, key=ga.calculate_fitness)

            min_distance = min(ga.calculate_fitness(route) for route in population)
            print(f"Epoch {epoch+1}: Minimum Distance = {1/min_distance:.2f} units")

    best_distance = sum(city.distance(best_route[(i+1) % num_cities]) for i, city in enumerate(best_route))

    print("\nUser Parameters:")
    print(f"Number of Cities: {num_cities}")
    print(f"Population Size: {population_size}")
    print(f"Mutation Rate: {mutation_rate}")
    print(f"Proportion of New Children: {new_children_proportion}")
    print(f"Using Stagnation: {'Yes' if use_stagnation else 'No'}")
    if use_stagnation:
        print(f"Stagnation Generations: {stagnation_generations}")
    else:
        print(f"Number of Generations: {num_generations}")

    print("\nBest sequence of cities:")
    for city in best_route:
        print(city.x, city.y)

    print(f"\nBest total distance: {best_distance:.2f} units")

if __name__ == "__main__":
    main()
