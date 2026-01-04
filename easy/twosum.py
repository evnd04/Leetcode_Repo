"""
LeetCode: Two Sum

Given an integer array `nums` and an integer `target`, return the indices of
two numbers such that they add up to `target`. You may assume exactly one
solution exists, and you cannot use the same element twice.

Approach
--------
Use a hash map (dictionary) to store numbers we've already seen and their
indices.

Algorithm
---------
- Initialize `num_map` as {value: index}.
- Iterate through `nums` with index `i` and value `x`:
  1) Compute `complement = target - x`.
  2) If `complement` is already in `num_map`, we found the pair:
     return `[num_map[complement], i]`.
  3) Otherwise, store the current value: `num_map[x] = i`.
- Return `[]` (only reached if no solution, though the problem guarantees one).

Cheat Sheet Items Used
----------------------
- Dictionaries (hash maps): `{}`, membership `key in dict`, insert/update
- Loops: `for i, x in enumerate(nums):`
- Lists: returning `[idx1, idx2]`

Complexity
----------
Time:  O(n) average (single pass, O(1) average hash lookups)
Space: O(n) for the hash map
"""

from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        num_map = {}

        for i, x in enumerate(nums):
            complement = target - x
            if complement in num_map:
                return [num_map[complement], i]
            num_map[x] = i

        return []
