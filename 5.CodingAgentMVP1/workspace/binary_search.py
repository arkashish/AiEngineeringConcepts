"""Binary Search Implementation.

This module provides a simple binary search function that searches for a target value
in a sorted list of comparable items and returns its index or ``-1`` if the
target is not present.

The implementation uses an iterative approach to avoid recursion depth
issues for large lists.

Example
-------
>>> from binary_search import binary_search
>>> binary_search([1, 3, 5, 7, 9], 5)
2
>>> binary_search([1, 3, 5, 7, 9], 2)
-1
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def binary_search(arr: Sequence[T], target: T) -> int:
    """Return the index of *target* in *arr* using binary search.

    Parameters
    ----------
    arr:
        A sorted sequence of comparable items.
    target:
        The value to search for.

    Returns
    -------
    int
        The index of the target in *arr* if it exists; otherwise ``-1``.
    """

    left: int = 0
    right: int = len(arr) - 1

    while left <= right:
        mid: int = (left + right) // 2
        mid_val: T = arr[mid]
        if mid_val == target:
            return mid
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


if __name__ == "__main__":
    import sys

    # Simple command line usage: python binary_search.py <sorted-list> <target>
    if len(sys.argv) < 3:
        print("Usage: python binary_search.py <sorted-list> <target>")
        sys.exit(1)

    # Convert first arg to list of ints
    try:
        arr = list(map(int, sys.argv[1].split(",")))
        target = int(sys.argv[2])
    except ValueError as e:
        print("Error parsing input:", e)
        sys.exit(1)

    print(f"Searching for {target} in {arr}")
    index = binary_search(arr, target)
    if index != -1:
        print(f"Found at index {index}")
    else:
        print("Target not found")
