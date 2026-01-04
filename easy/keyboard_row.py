"""
LeetCode: Keyboard Row

Given `words`, return the words that can be typed using letters from only ONE
row of the American keyboard (case-insensitive).

Approach
--------
- Build 3 `set`s for the keyboard rows (fast membership checks: `ch in row`).
- For each word:
  1) Convert to lowercase with `.lower()`.
  2) Pick the target row based on the first letter.
  3) Keep the word only if every character is in that same row.

Cheat Sheet Items Used
----------------------
- Sets: `set("qwertyuiop")`, membership check `ch in row`
- Strings: `.lower()`
- Loops: `for ... in ...`
- Built-ins: `all(...)`
- Lists: `ans = []`, `ans.append(w)`

Complexity
----------
Time:  O(N * L)  where N = number of words, L = max word length
Space: O(1)      (row sets are constant size)
"""

from typing import List

class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        row1 = set("qwertyuiop")
        row2 = set("asdfghjkl")
        row3 = set("zxcvbnm")

        ans = []
        for word in words:
            s = word.lower()

            if s[0] in row1:
                row = row1
            elif s[0] in row2:
                row = row2
            else:
                row = row3
            b = True

            for ch in s:
                if ch not in row:
                    b = False
                    break

            if b is True:
                ans.append(word)
                
        return ans