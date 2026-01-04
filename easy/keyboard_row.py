"""
LeetCode: Keyboard Row

Problem Description
-------------------
Given a list of strings `words`, return all words that can be typed using letters
from only one row of the American keyboard. The check is case-insensitive:
uppercase and lowercase versions of a letter are treated the same.

Keyboard rows:
    Row 1: "qwertyuiop"
    Row 2: "asdfghjkl"
    Row 3: "zxcvbnm"

A word is valid if every character in the word belongs to the same row.

Examples:
    Input:  ["Hello", "Alaska", "Dad", "Peace"]
    Output: ["Alaska", "Dad"]

    Input:  ["omk"]
    Output: []

    Input:  ["adsdf", "sfd"]
    Output: ["adsdf", "sfd"]


Solution Overview
-----------------
This solution uses set membership to quickly determine whether all letters in a
word come from a single keyboard row.

Algorithm Steps
---------------
1) Build three sets representing the letters in each keyboard row.
2) For each word:
   a) Convert it to lowercase for case-insensitive comparison.
   b) Identify the target row based on the word's first character.
   c) Check every character:
      - If any character is not in the target row, the word is invalid.
      - If all characters are in the target row, keep the original word.
3) Return the collected valid words.

Complexity
----------
Let N be the number of words and L be the maximum word length.
Time:  O(N * L)  (each character checked once)
Space: O(1)      (constant extra space; row sets are fixed size)
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