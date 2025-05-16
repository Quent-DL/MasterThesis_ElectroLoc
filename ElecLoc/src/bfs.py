"""A class for performing Breadth-First-Search used by the classification
module to quickly find the best set of models to fit the data."""


from typing import TypeAlias, Tuple, Callable, Iterable

from collections import deque
from itertools import combinations


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
                 goal_score: float,
                 max_n_children: int = 3):
        """TODO Write documentation.
        The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = State(tuple())
        self._candidates = set(candidates)

        self._func_scoring = scoring_function
        self._func_child_score = children_value_function
        self._goal_score = goal_score
        self._max_n_children = max_n_children

        # To cache the absolute score of the states
        self._cache_scores: dict[State, float] = {}

        # To cache the priority of the states, to quickly sort the children of
        # a state. 
        self._cache_child_prio: dict[State, float] = {}

    def children(self, state: State) -> list[State]:
        """Generic: Return the states that can be reached from the given
        state."""
        # Computing the available candidates for forming a new pair
        used_cand = {x for pair in state.pairs for x in pair}
        available_cand = self._candidates.difference(used_cand)

        # Computing all the possible childrens states and their scores
        scores: list[tuple[State, float]] = []
        for pair in combinations(available_cand, 2):
            child_state = State(state.pairs + (pair,))
            score = self.get_child_value(child_state)
            scores.append((child_state, score))

        # Returning only the best children
        scores.sort(key = lambda item: item[1], reverse=True)
        
        return [state for (state, score) in scores[:self._max_n_children]]

    def goal_test(self, state: State) -> bool:
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        return self.get_score(state) >= self._goal_score
    
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
        return [Node(state) for state in problem.children(self.state)]

    def is_better_than(self, 
                       other: 'Node', 
                       problem: MultimodelFittingProblem) -> bool:
        if other is None:
            return True
        
        better_score = (problem.get_score(self.state) 
                            > problem.get_score(other.state))
        self_goal = problem.goal_test(self.state)
        other_goal = problem.goal_test(other.state)

        depth_better = self.depth < other.depth
        depth_equal = self.depth == other.depth
        depth_worse = self.depth > other.depth
        
        return (
            (depth_better and (better_score or self_goal))
            or (depth_equal and better_score)
            or (depth_worse and self_goal and not other_goal)
        )


def breadth_first_graph_search(problem: MultimodelFittingProblem):
    """
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    # A limitation to stop the search
    max_depth = 100000000

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

                    # To prevent new solutions from being
                    # deeper than this child
                    if problem.goal_test(child.state):
                        max_depth = min(max_depth, child.depth)

                if child.depth < max_depth:
                    # Prevent this child from producing other children outside
                    # the depth range
                    frontier.append(child)

    assert best_node is not None
    return best_node.state.pairs