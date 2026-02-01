"""
LeetCode: Maximum Product Subarray

Given an integer array `nums`, return the maximum product of a contiguous
subarray.

Key Challenge
-------------
Unlike sums, products flip sign:
- A large negative times a negative can become a large positive.
So we must track BOTH:
- `curMax`: max product ending at current index
- `curMin`: min product ending at current index (most negative)

Approach
--------
Dynamic programming (1 pass), tracking running max/min products.

At each number `n`, the best/worst product ending here can be:
1) Start fresh at n           -> n
2) Extend previous curMax     -> curMax * n
3) Extend previous curMin     -> curMin * n  (important when n is negative)

So:
- newCurMax = max(n, curMax*n, curMin*n)
- newCurMin = min(n, curMax*n, curMin*n)

We keep a global answer `res` updated with the best `curMax` seen.

Why `temp`?
-----------
When updating `curMin`, we still need the OLD `curMax` value.
So we store `temp = curMax * n` before changing `curMax`.

Zeros
-----
If n == 0:
- Both curMax and curMin become 0 (since max/min among {0, 0, 0} is 0),
effectively "resetting" the product chain after zero.

Cheat Sheet Items Used
----------------------
- Built-ins: `max(...)`, `min(...)`
- Initialization: `res = max(nums)` to handle all-negative cases
- Running state variables: `curMax`, `curMin`
- Loop: `for n in nums:`
- Temporary variable: `temp` to preserve old state during updates

Complexity
----------
Time:  O(n)
Space: O(1)
"""

from typing import List

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        # Start with the best single element (important if all values are negative)
        res = max(nums)

        # curMax/curMin represent max/min product of a subarray ending at current index
        curMin = 1
        curMax = 1

        for n in nums:
            # Save old curMax*n before curMax gets overwritten
            temp = curMax * n

            # Compute new max/min products ending here
            curMax = max(temp, n * curMin, n)
            curMin = min(temp, n * curMin, n)

            # Update global best
            res = max(res, curMax)

        return res
