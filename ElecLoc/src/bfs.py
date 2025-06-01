"""A class for performing Breadth-First-Search used by the modelling
module to quickly find the best set of models to fit the data.

Adapted from the code provided by Yves Deville 
for the projects of the course "LINFO1361 - Intelligence Artificielle".

Université Catholique de Louvain

2022-2023.
"""

from typing import TypeAlias, Tuple, Callable, Iterable

from collections import deque
from itertools import combinations
import numpy as np

Pair: TypeAlias= Tuple[int, int]


class State:
    def __init__(self, pairs: Pair):
        # Sorting pairs
        self.pairs = tuple(sorted(
            [tuple(sorted(pair)) for pair in pairs],
            key = lambda p: p[0]))

    def __str__(self):
        print(self.pairs)
    
    def __eq__(self, other) -> bool:
        """"Checks the equivalence of two states, only taking account of the position of the tiles"""
        if (not isinstance(other, State)):
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.pairs)
    
    
class MultimodelFittingProblem:
    N_SAMPLES_MOV_AV = 10

    def __init__(self,
                 candidates: Iterable[int],
                 scoring_function: Callable[[Tuple[Tuple[int, int]]], float],
                 children_value_function: Callable[[Tuple[Tuple[int, int]]], float],    # TODO test keep or delete
                 goal_depth: int,
                 tags_dcc: np.ndarray[int],
                 max_n_children: int = 2):
        """TODO Write documentation.
        The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = State(tuple())
        self._candidates = set(candidates)

        self._func_scoring = scoring_function
        self._func_child_score = children_value_function
        self._goal_depth = goal_depth
        self._max_n_children = max_n_children

        # Separating the candidates into sets according to the DCC id.
        # Only candidates from the same set can form a pair.
        tags_dcc -= 1    # Because tags_dcc initially contains {1, ..., D}
                         # but we want {0, ..., D-1}, to represent indices
        nb_dcc = len(np.unique(tags_dcc))
        self._init_cand_sets = [set() for _ in range(nb_dcc)]
        for cand in candidates:
            self._init_cand_sets[tags_dcc[cand]].add(cand)
        # Opti: sorting sets by size to handle small sets first 
        # (restricts size of graph)
        self._init_cand_sets.sort(key= lambda cand_set: len(cand_set))

        # To cache the absolute score of the states
        self._cache_scores: dict[State, float] = {}

        # To cache the priority of the states, to quickly sort the children of
        # a state. 
        self._cache_child_prio: dict[State, float] = {}

    def children(self, state: State) -> list[State]:
        """Generic: Return the states that can be reached from the given
        state."""

        # TODO init parameter max nb of children as a list [3,3,3,2,2,1,1,1] based on depth. 
        # 3 children at depths [0, 1, 2], 2 children at depths [3, 4], 1 child for depths >= 5
        n_children = self._max_n_children

        # Computing the available candidates for forming a new pair
        if self.goal_test(state):
            return []     # goal depth is last depth
        
        # Computing the unused candidates in each set
        used_cand = {x for pair in state.pairs for x in pair}
        sets_available_cand = [cand_set.difference(used_cand) 
                                   for cand_set in self._init_cand_sets]

        # Opti: if a DCC can only fit one electrode: handle that DCC has soon 
        # as possible -> force BFS to expand it
        forced = False
        for av_cand_set, init_can_set in zip(
                sets_available_cand, self._init_cand_sets):
            if (len(init_can_set) <= 3     # DCC contains only one electrode
                    and len(av_cand_set) >= 2):    # and it hasn't been treated yet
                # Force BFS to handle this electrode/DCC
                available_pairs = combinations(av_cand_set, 2)
                forced = True
                break

        if not forced:
            # Use all possible pairs if no DCC is forced to be handled
            available_pairs = set()
            for cand_set in sets_available_cand:
                available_pairs = available_pairs.union(combinations(cand_set, 2))

        # Computing all the possible childrens states and their scores
        scores: list[tuple[State, float]] = []
        for pair in available_pairs:
            child_state = State(state.pairs + (pair,))
            score = self.get_child_value(child_state)
            scores.append((child_state, score))

        # Returning only the best children
        scores.sort(key = lambda item: item[1], reverse=True)
        
        return [state for (state, score) in scores[:n_children]]

    def goal_test(self, state: State) -> bool:
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        return len(state.pairs) >= self._goal_depth
    
    def get_score(self, state: State) -> float:
        if state in self._cache_scores:
            score = self._cache_scores[state]
        else:
            score = self._func_scoring(state.pairs)
            self._cache_scores[state] = score
        return score
    
    def get_child_value(self, state: State) -> float:
        if state in self._cache_child_prio:
            val = self._cache_child_prio[state]
        else:
            val = self._func_child_score(state.pairs)
            self._cache_child_prio[state] = val
        return val
    
    def get_number_group_scores_computed(self) -> int:
        return len(self._cache_scores)


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, 
                 state: State, 
                 parent: 'Node' = None):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node: 'Node'):
        return self.state < node.state

    def expand(self, problem: MultimodelFittingProblem) -> list['Node']:
        """List the nodes reachable in one step from this node."""
        return [Node(state, self) for state in problem.children(self.state)]

    def is_better_than(self, 
                       other: 'Node', 
                       problem: MultimodelFittingProblem) -> bool:
        if other is None:
            return True
        
        better_score = (problem.get_score(self.state) 
                            > problem.get_score(other.state))
        self_goal = problem.goal_test(self.state)
        other_goal = problem.goal_test(other.state)
        
        return (self_goal and not other_goal) or better_score

def breadth_first_graph_search(problem: MultimodelFittingProblem):
    """
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """

    # The result of the search
    best_node: Node = None

    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    
    frontier = deque([node])
    explored = set()

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)

        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:

                if child.is_better_than(best_node, problem):
                    # Updating best found yet
                    best_node = child

                if not problem.goal_test(child.state):
                    # Preventing this child from producing other children 
                    # outside the depth upper bound
                    frontier.append(child)

    assert best_node is not None
    return best_node.state.pairs