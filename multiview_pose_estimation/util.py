import numpy as np


def joints(parents: np.ndarray) -> np.ndarray:
    """Return an array of joint indices.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        An array of joint indices. Shape: (J,)
    """
    return np.arange(len(parents), dtype=int)


def joints_list(parents: np.ndarray) -> list[np.ndarray]:
    """Return a list of arrays of joint indices for each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A list of arrays of joint indices. Shape: [(1, J)]
    """
    return list(joints(parents)[:, np.newaxis])


def parents_list(parents: np.ndarray) -> list[np.ndarray]:
    """Return a list of arrays of joint indices for the parents of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A list of arrays of joint indices for the parents of each joint. Shape: [(1, J)]
    """
    return list(parents[:, np.newaxis])


def children_list(parents: np.ndarray) -> list[np.ndarray]:
    """Return a list of arrays of joint indices for the children of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A list of arrays of joint indices for the children of each joint. Shape: [(C1,), (C2,), ... , (CJ,)]
    """

    def joint_children(i: int) -> list[int]:
        return [j for j, p in enumerate(parents) if p == i]

    return list(map(lambda j: np.array(joint_children(j)), joints(parents)))


def descendants_list(parents: np.ndarray) -> list[np.ndarray]:
    """Return a list of arrays of joint indices for the descendants of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A list of arrays of joint indices for the descendants of each joint. Shape: [(D1,), (D2,), ... , (DJ,)]
    """

    children = children_list(parents)

    def joint_descendants(i: int) -> list[int]:
        return sum([joint_descendants(j) for j in children[i]], list(children[i]))

    return list(map(lambda j: np.array(joint_descendants(j)), joints(parents)))


def ancestors_list(parents: np.ndarray) -> list[np.ndarray]:
    """Return a list of arrays of joint indices for the ancestors of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A list of arrays of joint indices for the ancestors of each joint. Shape: [(A1,), (A2,), ... , (AJ,)]
    """

    descendants = descendants_list(parents)

    def joint_ancestors(i: int) -> list[int]:
        return [j for j in joints(parents) if i in descendants[j]]

    return list(map(lambda j: np.array(joint_ancestors(j)), joints(parents)))

def mask(parents: np.ndarray, filter: callable[[np.ndarray], list[np.ndarray]]) -> np.ndarray:
    """Construct a mask for a given filter.

    A mask is a boolean truth table of size (J, J) ndarray for a given
    condition over J joints. For example there may be a mask specifying
    if a joint N is a child of another joint M. This could be constructed
    into a mask using `m = mask(parents, children_list)` and the condition
    of childhood tested using `m[N, M]`.

    Args:
        parents: An array of parents. Shape: (J,)
        filter: A function that outputs a list of arrays of joint indices for some condition.

    Returns:
        A boolean truth table of given condition. Shape: (J, J)
    """
    m = np.zeros((len(parents), len(parents))).astype(bool)
    jnts = joints(parents)
    fltr = filter(parents)
    for i, f in enumerate(fltr):
        m[i, :] = np.any(jnts[:, np.newaxis] == f[np.newaxis, :], axis=1)
    return m


def joints_mask(parents: np.ndarray) -> np.ndarray:
    """Return a boolean truth table of joint indices.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A boolean truth table of joint indices. Shape: (J, J)
    """
    return np.eye(len(parents)).astype(bool)


def children_mask(parents: np.ndarray) -> np.ndarray:
    """Return a boolean truth table of joint indices for the children of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A boolean truth table of joint indices for the children of each joint. Shape: (J, J)
    """
    return mask(parents, children_list)


def parents_mask(parents: np.ndarray) -> np.ndarray:
    """Return a boolean truth table of joint indices for the parents of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A boolean truth table of joint indices for the parents of each joint. Shape: (J, J)
    """
    return mask(parents, parents_list)


def descendants_mask(parents: np.ndarray) -> np.ndarray:
    """Return a boolean truth table of joint indices for the descendants of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A boolean truth table of joint indices for the descendants of each joint. Shape: (J, J)
    """
    return mask(parents, descendants_list)


def ancestors_mask(parents: np.ndarray) -> np.ndarray:
    """Return a boolean truth table of joint indices for the ancestors of each joint.

    Args:
        parents: An array of parents. Shape: (J,)

    Returns:
        A boolean truth table of joint indices for the ancestors of each joint. Shape: (J, J)
    """
    return mask(parents, ancestors_list)
