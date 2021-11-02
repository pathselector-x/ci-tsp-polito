# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

NUM_CITIES = 42
STEADY_STATE = 200

class Tsp:

    def __init__(self, num_cities: int, seed: Any = None) -> None:
        if seed is None:
            seed = num_cities
        self._num_cities = num_cities
        self._graph = nx.DiGraph()
        np.random.seed(seed)
        for c in range(num_cities):
            self._graph.add_node(c, pos=(np.random.random(), np.random.random()))

    def distance(self, n1, n2) -> int:
        pos1 = self._graph.nodes[n1]['pos']
        pos2 = self._graph.nodes[n2]['pos']
        return round(1_000_000 / self._num_cities * sqrt((pos1[0] - pos2[0])**2 +
                                                         (pos1[1] - pos2[1])**2))

    def evaluate_solution(self, solution: np.array) -> float:
        total_cost = 0
        tmp = solution.tolist() + [solution[0]]
        for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
            total_cost += self.distance(n1, n2)
        return total_cost

    def plot(self, path: np.array = None) -> None:
        if path is not None:
            self._graph.remove_edges_from(list(self._graph.edges))
            tmp = path.tolist() + [path[0]]
            for n1, n2 in (tmp[i:i + 2] for i in range(len(tmp) - 1)):
                self._graph.add_edge(n1, n2)
        plt.figure(figsize=(12, 5))
        nx.draw(self._graph,
                pos=nx.get_node_attributes(self._graph, 'pos'),
                with_labels=True,
                node_color='pink')
        if path is not None:
            plt.title(f"Current path: {self.evaluate_solution(path):,}")
        plt.show()

    @property
    def graph(self) -> nx.digraph:
        return self._graph

def tweak(problem, path, temperature, t_pow=1.0, dist_pow=3.0):

    #! We need to select next city in some way
    probability = 0.0
    total_distance = 0.0

    not_visited = [city for city in range(NUM_CITIES) if city not in path]
    for city in not_visited:
        total_distance += problem.distance(path[-1], city)
    for city in not_visited:
        probability += temperature[path[-1], city]**t_pow * (total_distance / problem.distance(path[-1], city))**dist_pow
    
    pm = np.random.uniform(0.0, probability)
    p = 0.0
    for city in not_visited:
        p += temperature[path[-1], city]**t_pow * (total_distance / problem.distance(path[-1], city))**dist_pow
        if p >= pm:
            return city

def main():

    POPULATION_SIZE = 5

    T_DECAY = 0.8
    T_POW = 1.0
    T_ADD = 1.0
    DIST_POW = 3.0

    problem = Tsp(NUM_CITIES)

    temperature = np.ones((NUM_CITIES, NUM_CITIES), dtype=np.float32)

    best_path = None
    best_distance = np.inf

    steady_state = 0
    step = 0

    while steady_state < STEADY_STATE:
        steady_state += 1
        step += 1
        for individual in range(POPULATION_SIZE):

            #! Find path
            path = [np.random.randint(0, NUM_CITIES)]
            while len(path) < NUM_CITIES:
                path.append(tweak(problem, path, temperature, t_pow=T_POW, dist_pow=DIST_POW))
            path = np.array(path)

            #! Compute distance
            distance = problem.evaluate_solution(path)

            #! Add temperature
            temp_add = 1.0 / distance
            for i in range(NUM_CITIES):
                temperature[path[i], path[(i+1) % NUM_CITIES]] += T_ADD * temp_add

            #! Check if better
            if distance < best_distance:
                best_distance = distance
                best_path = path

                steady_state = 0
        
        #! Add temperature
        temp_add = 1.0 / best_distance
        for i in range(NUM_CITIES):
            temperature[best_path[i], best_path[(i+1) % NUM_CITIES]] += T_ADD * temp_add
            
        #! Cool down temperature
        for i in range(NUM_CITIES):
            for j in range(NUM_CITIES):
                temperature[i,j] *= T_DECAY

    print(f'\nBest distance found: {best_distance:,} in {step} steps\n')
    problem.plot(best_path)

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()