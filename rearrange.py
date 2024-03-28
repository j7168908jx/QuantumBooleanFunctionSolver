from typing import List
import numpy as np

from utils import _memorize

@_memorize
def rearrange(n: int, m: int, works: List[List[int]]) -> List[int]:
    """Find the greedy rearrange order to reduce circuit depth.

    The circuit optimization problem is abstracted as a scheduling problem.

    Several `work` is defined in the `works` list. Design a parallel schedule
    such that more works can be done at the same time.
    Each work requires 2 or 3 nodes to work together for 1 time period and
    consists at least 1 node in nodes set 1 and exactly 1 node in nodes set 2

    Args:
        n: num of nodes in set 1
        m: num of nodes in set 2
        works: a list of all works, each work use 2 or 3 integers
            to represent required nodes (0, ..., m-1 for nodes in set 1 and
            m, ... m+n-1 for nodes in set 2)

    Returns:
        Rearrnged work order, a shuffled list of `[0, ..., len(works)-1]`

    Examples:
        Input: 4, 2, [[1,3,5], [2,5], [2,5], [1,3,4], [1,4]]

        Interpretation: There are 4 nodes in nodes set 1 and 2 in nodes set 2.
        The first work requires node 1,3,5 to work together (node 5 means
        the first node in nodes set 2)

        Now the time cost is 4, since only the 3rd and 4th work can be done
        at the same time.

        If we give rearranged order [0, 1, 3, 4, 2]:
            [[1,3,5], [2,5], [2,5], [1,3,4], [1,4]] ->
            [[1,3,5], [2,5], [1,3,4], [1,4], [2,5]]
        Then the time cost becomes 3.

    """
    works_with_order_num = [(i, w) for i, w in enumerate(works)]
    result = []
    while works_with_order_num:
        nodes = np.zeros((m+n,))
        for i, w in enumerate(works_with_order_num):
            if not np.any(nodes[w[1]] != 0):
                for node_num in w[1]:
                    nodes[node_num] = 1
                result.append(w[0])
                works_with_order_num.pop(i)
    return result
