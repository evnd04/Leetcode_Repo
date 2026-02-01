"""
LeetCode: Product of Array Except Self

Given an integer array `nums`, return an array `res` such that:
    res[i] = product of all nums[j] for j != i

Constraints/notes:
- Do NOT use division.
- Must run in O(n) time.
- Extra space should be O(1) (output array `res` does not count).

Approach
--------
Two-pass prefix/postfix products.

Idea:
- If we knew:
    prefix_product_before_i  = nums[0] * ... * nums[i-1]
    postfix_product_after_i  = nums[i+1] * ... * nums[n-1]
  then:
    res[i] = prefix_product_before_i * postfix_product_after_i

We compute these without extra arrays:
1) Left-to-right pass:
   - Maintain `prefix` = product of elements seen so far (to the LEFT of i).
   - Set res[i] = prefix, then update prefix *= nums[i].

2) Right-to-left pass:
   - Maintain `postfix` = product of elements seen so far (to the RIGHT of i).
   - Multiply res[i] *= postfix, then update postfix *= nums[i].

This naturally handles zeros:
- If there is one zero, only the position of the zero gets the product of non-zero elements.
- If there are two+ zeros, everything becomes 0.

Cheat Sheet Items Used
----------------------
- Lists: `res = [1] * len(nums)`
- Loops: `for i in range(...)`, reverse range `range(n-1, -1, -1)`
- Variables as running products: `prefix`, `postfix`
- In-place update of output array: `res[i] *= postfix`

Complexity
----------
Time:  O(n)   (two linear passes)
Space: O(1)   extra space (excluding output array `res`)
"""

from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # res[i] will eventually hold:
        #   (product of elements left of i) * (product of elements right of i)
        res = [1] * len(nums)

        # PASS 1 (prefix products):
        # prefix = product of nums[0..i-1] before processing index i
        prefix = 1
        for i in range(len(nums)):
            res[i] = prefix
            prefix *= nums[i]

        # PASS 2 (postfix products):
        # postfix = product of nums[i+1..n-1] before processing index i
        postfix = 1
        for i in range(len(nums) - 1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]

        return res
