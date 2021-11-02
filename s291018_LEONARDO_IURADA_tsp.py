# Copyright © 2021 Giovanni Squillero <squillero@polito.it>
# Free for personal or classroom use; see 'LICENCE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
from math import sqrt
from typing import Any
from typing_extensions import final
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import deque

NUM_CITIES = 23
STEADY_STATE = 1000

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

def tweak(solution: np.array, *, pm: float = .1) -> np.array:
    new_solution = solution.copy()
    p = None
    while p is None or p < pm:
        i1 = np.random.randint(0, solution.shape[0])
        i2 = np.random.randint(0, solution.shape[0])
        temp = new_solution[i1]
        new_solution[i1] = new_solution[i2]
        new_solution[i2] = temp
        p = np.random.random()
    return new_solution

def main():

    problem = Tsp(NUM_CITIES)

    num_instances = 1000

    #problem.plot(solution)
#    steady_state = 0
#    step = 0
#    while steady_state < STEADY_STATE:
#        step += 1
#        steady_state += 1
#        new_solution = tweak(solution, pm=.5)
#        new_solution_cost = problem.evaluate_solution(new_solution)
#        if new_solution_cost < solution_cost:
#            solution = new_solution
#            solution_cost = new_solution_cost
#            history.append((step, solution_cost))
#            steady_state = 0

    DES_POW = 2.6
    PHERO_POW = 10

    final_solution = None
    final_sol_cost = np.inf

    phero = np.array([0 for 0 in range(NUM_CITIES)])

    for _ in range(num_instances):

        solution = np.array(range(NUM_CITIES))
        np.random.shuffle(solution)
        solution_cost = problem.evaluate_solution(solution)

        new_solution = np.array([solution[0]])
        solution = solution[1:]

        current_city = new_solution[-1]

        while len(solution):
            distances = np.array([problem.distance(current_city, city) for city in solution], dtype=np.float32)

            desirability = np.array([(1 / d)**DES_POW for d in distances], dtype=np.float32) 
            phero = np.array()

            # We weight based on desirability and pheromone
            distances *= desirability

            norm_dist = distances.sum()
            probabilities = np.array([d / norm_dist for d in distances], dtype=np.float32)

            next_city = np.random.choice(solution, 1, p=probabilities)
            new_solution = np.append(new_solution, next_city)

            solution = np.delete(solution, np.where(solution == next_city))

            current_city = next_city[0]

        new_sol_cost = problem.evaluate_solution(new_solution) 

        if final_solution is None or new_sol_cost < final_sol_cost:
            final_solution = new_solution.copy()
            final_sol_cost = new_sol_cost

    print(final_sol_cost)
    problem.plot(final_solution)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().setLevel(level=logging.INFO)
    main()