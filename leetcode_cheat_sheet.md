# LeetCode Cheat Sheet

## Table of Contents
1. [Python Basics](#python-basics)
2. [Data Structures](#data-structures)
3. [Common Algorithms & Patterns](#common-algorithms--patterns)
4. [Time & Space Complexity](#time--space-complexity)

---

## Python Basics

### Variables & Types
```python
# Basic types
x = 5              # int
y = 3.14           # float
z = "hello"        # str
b = True           # bool
n = None           # NoneType

# Type conversion
int("123")         # 123
str(123)           # "123"
float("3.14")      # 3.14
bool(1)            # True
```

### Lists (Arrays)
```python
# Creation
arr = [1, 2, 3, 4, 5]
arr = list(range(5))  # [0, 1, 2, 3, 4]
arr = [0] * 5         # [0, 0, 0, 0, 0]

# Accessing
arr[0]              # First element
arr[-1]             # Last element
arr[1:3]            # Slice [1, 2]
arr[:3]             # First 3 elements
arr[2:]             # From index 2 to end

# Common operations
arr.append(6)       # Add to end
arr.insert(0, 0)    # Insert at index
arr.pop()           # Remove last
arr.pop(0)          # Remove at index
arr.remove(3)       # Remove first occurrence
arr.index(3)        # Find index of value
arr.count(3)        # Count occurrences
arr.reverse()       # Reverse in place
arr.sort()          # Sort in place
sorted(arr)         # Return sorted copy
len(arr)            # Length

# List comprehension
[x*2 for x in range(5)]              # [0, 2, 4, 6, 8]
[x for x in range(10) if x % 2 == 0] # [0, 2, 4, 6, 8]
```

### Strings
```python
# Creation
s = "hello"
s = 'world'
s = """multi
line"""

# Common operations
s[0]                # 'h'
s[-1]               # 'o'
s[1:4]              # 'ell'
s.upper()           # 'HELLO'
s.lower()           # 'hello'
s.strip()           # Remove whitespace
s.split(',')        # Split by delimiter
s.replace('l', 'L') # Replace characters
s.find('e')         # Find index, -1 if not found
s.index('e')        # Find index, raises if not found
s.startswith('he')  # True
s.endswith('lo')    # True
s.isdigit()         # Check if all digits
s.isalpha()         # Check if all letters
s.isalnum()         # Check if alphanumeric
''.join(['a','b'])  # 'ab'
len(s)              # Length

# String formatting
f"Value: {x}"       # f-string (Python 3.6+)
"Value: {}".format(x)
"Value: %d" % x
```

### Dictionaries (Hash Maps)
```python
# Creation
d = {}
d = {'a': 1, 'b': 2}
d = dict(a=1, b=2)

# Accessing
d['a']              # 1
d.get('a', 0)       # 1, or 0 if key doesn't exist
d.get('c', 0)       # 0 (safe access)

# Common operations
d['c'] = 3          # Add/update
d.pop('a')          # Remove and return value
d.popitem()         # Remove and return last item
del d['b']          # Delete key
'a' in d            # Check if key exists
d.keys()            # All keys
d.values()          # All values
d.items()           # All (key, value) pairs
len(d)              # Number of items

# Dictionary comprehension
{x: x*2 for x in range(5)}  # {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}
```

### Sets
```python
# Creation
s = set()
s = {1, 2, 3}
s = set([1, 2, 3])

# Common operations
s.add(4)            # Add element
s.remove(3)         # Remove, raises if not found
s.discard(3)        # Remove, no error if not found
s.pop()             # Remove and return arbitrary element
3 in s              # Check membership
len(s)              # Size

# Set operations
s1 | s2             # Union
s1 & s2             # Intersection
s1 - s2             # Difference
s1 ^ s2             # Symmetric difference

# Set comprehension
{x for x in range(10) if x % 2 == 0}  # {0, 2, 4, 6, 8}
```

### Tuples
```python
# Creation
t = (1, 2, 3)
t = 1, 2, 3         # Parentheses optional
t = tuple([1, 2, 3])

# Accessing
t[0]                # 1
t[-1]               # 3
t[1:3]              # (2, 3)

# Common operations
len(t)              # Length
t.count(2)          # Count occurrences
t.index(2)          # Find index
```

### Control Flow
```python
# If/Else
if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")

# For loops
for i in range(5):          # 0 to 4
    print(i)

for i in range(1, 6):       # 1 to 5
    print(i)

for i in range(0, 10, 2):   # 0, 2, 4, 6, 8
    print(i)

for item in arr:
    print(item)

for i, item in enumerate(arr):
    print(i, item)

for key, value in d.items():
    print(key, value)

# While loops
while x > 0:
    x -= 1

# Break and continue
for i in range(10):
    if i == 5:
        break       # Exit loop
    if i % 2 == 0:
        continue    # Skip to next iteration
```

### Functions
```python
# Basic function
def add(a, b):
    return a + b

# Default arguments
def greet(name="World"):
    return f"Hello, {name}"

# Variable arguments
def sum_all(*args):
    return sum(args)

def print_kwargs(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Lambda functions
square = lambda x: x * x
add = lambda x, y: x + y

# Map, filter, reduce
map(lambda x: x*2, [1,2,3])         # [2, 4, 6]
filter(lambda x: x > 2, [1,2,3,4])  # [3, 4]
from functools import reduce
reduce(lambda x, y: x + y, [1,2,3]) # 6
```

### Classes
```python
class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    def __repr__(self):
        return f"Node({self.val})"

# Inheritance
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Useful Built-in Functions
```python
# Math
abs(-5)             # 5
max(1, 2, 3)        # 3
min(1, 2, 3)        # 1
sum([1, 2, 3])      # 6
pow(2, 3)           # 8
round(3.14159, 2)   # 3.14

# Type checking
type(x)             # Get type
isinstance(x, int)  # Check type

# Collections
all([True, True, False])    # False
any([True, False, False])   # True
enumerate(['a', 'b'])       # [(0, 'a'), (1, 'b')]
zip([1, 2], ['a', 'b'])     # [(1, 'a'), (2, 'b')]

# Sorting
sorted([3, 1, 2])           # [1, 2, 3]
sorted([3, 1, 2], reverse=True)  # [3, 2, 1]
sorted([('a', 3), ('b', 1)], key=lambda x: x[1])  # [('b', 1), ('a', 3)]
```

### Collections Module
```python
from collections import defaultdict, deque, Counter, OrderedDict

# Defaultdict
dd = defaultdict(int)       # Default value is 0
dd = defaultdict(list)      # Default value is []

# Deque (double-ended queue)
dq = deque()
dq.append(1)                # Add to right
dq.appendleft(2)            # Add to left
dq.pop()                    # Remove from right
dq.popleft()                # Remove from left

# Counter
c = Counter([1, 2, 2, 3])   # {1: 1, 2: 2, 3: 1}
c.most_common(2)            # [(2, 2), (1, 1)]
```

### Heap (Priority Queue)
```python
import heapq

# Min heap (default)
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)
heapq.heappop(heap)         # 1 (smallest)

# Max heap (negate values)
heap = []
heapq.heappush(heap, -3)
heapq.heappush(heap, -1)
heapq.heappush(heap, -2)
-heapq.heappop(heap)        # 3 (largest)

# Heapify existing list
arr = [3, 1, 2]
heapq.heapify(arr)
```

### Useful Tips
```python
# Unpacking
a, b = [1, 2]
a, *rest = [1, 2, 3, 4]     # a=1, rest=[2,3,4]

# Swapping
a, b = b, a

# Multiple assignment
x = y = z = 0

# Check if empty
if not arr:                 # Empty list
    pass

# String to list and back
list("hello")               # ['h', 'e', 'l', 'l', 'o']
''.join(['h', 'e', 'l', 'l', 'o'])  # "hello"
```

---

## Data Structures

### Arrays/Lists
- **Access**: O(1)
- **Search**: O(n)
- **Insert/Delete at end**: O(1)
- **Insert/Delete at middle**: O(n)

```python
arr = [1, 2, 3, 4, 5]
```

### Linked Lists
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Common operations
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev

def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### Stacks
```python
# Using list
stack = []
stack.append(1)     # Push
stack.pop()         # Pop
stack[-1]           # Peek (if not empty)

# Common pattern: Monotonic stack
def next_greater_element(nums):
    stack = []
    result = [-1] * len(nums)
    for i, num in enumerate(nums):
        while stack and nums[stack[-1]] < num:
            result[stack.pop()] = num
        stack.append(i)
    return result
```

### Queues
```python
from collections import deque

queue = deque()
queue.append(1)     # Enqueue
queue.popleft()     # Dequeue
queue[0]            # Peek (if not empty)
```

### Trees
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Tree Traversals
def inorder(root):
    if not root:
        return
    inorder(root.left)
    print(root.val)
    inorder(root.right)

def preorder(root):
    if not root:
        return
    print(root.val)
    preorder(root.left)
    preorder(root.right)

def postorder(root):
    if not root:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)

# Level-order (BFS)
from collections import deque
def level_order(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### Graphs
```python
# Adjacency List representation
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}

# DFS
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

# BFS
from collections import deque
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
```

### Hash Tables (Dictionaries)
- **Average**: O(1) for all operations
- **Worst**: O(n) for all operations (collision)

```python
hash_map = {}
hash_map[key] = value
value = hash_map.get(key, default)
```

### Heaps (Priority Queues)
- **Insert**: O(log n)
- **Extract min/max**: O(log n)
- **Peek**: O(1)

```python
import heapq
# See Python Basics section for examples
```

### Trie (Prefix Tree)
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### Union-Find (Disjoint Set)
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
```

---

## Common Algorithms & Patterns

### 1. Two Pointers
```python
# Array sorted, find two sum
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current_sum = arr[left] + arr[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

# Remove duplicates
def remove_duplicates(nums):
    if not nums:
        return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

### 2. Sliding Window
```python
# Fixed window size
def max_sum_subarray(arr, k):
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        window_sum = window_sum - arr[i-k] + arr[i]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Variable window size
def longest_substring_without_repeating(s):
    char_map = {}
    left = 0
    max_len = 0
    for right in range(len(s)):
        if s[right] in char_map:
            left = max(left, char_map[s[right]] + 1)
        char_map[s[right]] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```

### 3. Binary Search
```python
# Standard binary search
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Find first occurrence
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return result

# Search in rotated array
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

### 4. Depth-First Search (DFS)
```python
# Recursive DFS
def dfs_recursive(node, visited):
    if node in visited:
        return
    visited.add(node)
    # Process node
    for neighbor in node.neighbors:
        dfs_recursive(neighbor, visited)

# Iterative DFS
def dfs_iterative(start):
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        # Process node
        for neighbor in node.neighbors:
            if neighbor not in visited:
                stack.append(neighbor)
```

### 5. Breadth-First Search (BFS)
```python
from collections import deque

def bfs(start):
    queue = deque([start])
    visited = set([start])
    while queue:
        node = queue.popleft()
        # Process node
        for neighbor in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### 6. Dynamic Programming
```python
# Fibonacci (memoization)
def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]

# Fibonacci (tabulation)
def fib_tabulation(n):
    if n <= 2:
        return 1
    dp = [0] * (n + 1)
    dp[1] = dp[2] = 1
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w-weights[i-1]] + values[i-1]
                )
            else:
                dp[i][w] = dp[i-1][w]
    return dp[n][capacity]

# Longest Common Subsequence
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### 7. Backtracking
```python
def backtrack(candidate, path, result):
    if is_solution(candidate):
        result.append(path[:])  # Make a copy
        return
    
    for next_candidate in get_candidates(candidate):
        path.append(next_candidate)
        backtrack(next_candidate, path, result)
        path.pop()  # Backtrack

# Example: Generate all permutations
def permute(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            backtrack(path + [remaining[i]], remaining[:i] + remaining[i+1:])
    backtrack([], nums)
    return result

# Example: N-Queens
def solve_n_queens(n):
    result = []
    board = [['.'] * n for _ in range(n)]
    
    def is_safe(row, col):
        for i in range(row):
            if board[i][col] == 'Q':
                return False
            if col - (row - i) >= 0 and board[i][col - (row - i)] == 'Q':
                return False
            if col + (row - i) < n and board[i][col + (row - i)] == 'Q':
                return False
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(row) for row in board])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    backtrack(0)
    return result
```

### 8. Greedy Algorithms
```python
# Activity Selection
def activity_selection(activities):
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    for activity in activities[1:]:
        if activity[0] >= selected[-1][1]:
            selected.append(activity)
    return selected

# Interval Scheduling
def erase_overlap_intervals(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    count = 0
    end = intervals[0][1]
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            count += 1
        else:
            end = intervals[i][1]
    return count
```

### 9. Topological Sort
```python
from collections import deque

def topological_sort(graph):
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            in_degree[neighbor] += 1
    
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == len(graph) else []  # Check for cycle
```

### 10. Monotonic Stack
```python
# Next Greater Element
def next_greater_element(nums):
    stack = []
    result = [-1] * len(nums)
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result

# Largest Rectangle in Histogram
def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    while stack:
        h = heights[stack.pop()]
        w = len(heights) if not stack else len(heights) - stack[-1] - 1
        max_area = max(max_area, h * w)
    return max_area
```

### 11. Fast & Slow Pointers (Floyd's Cycle Detection)
```python
# Detect cycle in linked list
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Find cycle start
def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow
```

### 12. Merge Intervals
```python
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    return merged
```

### 13. K-way Merge
```python
import heapq

def merge_k_sorted_arrays(arrays):
    heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    
    result = []
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(arrays[arr_idx]):
            heapq.heappush(heap, (arrays[arr_idx][elem_idx + 1], arr_idx, elem_idx + 1))
    return result
```

### 14. Subset Generation
```python
# Generate all subsets
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result

# Generate subsets with duplicates
def subsets_with_dup(nums):
    nums.sort()
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result
```

### 15. String Matching (KMP Algorithm)
```python
def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps
    
    lps = build_lps(pattern)
    i = j = 0
    result = []
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == len(pattern):
            result.append(i - j)
            j = lps[j - 1]
        elif i < len(text) and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return result
```

---

## Time & Space Complexity

### Big O Notation Cheat Sheet

#### Time Complexities
- **O(1)**: Constant time - accessing array element, hash table lookup
- **O(log n)**: Logarithmic - binary search, heap operations
- **O(n)**: Linear - single loop through array
- **O(n log n)**: Linearithmic - efficient sorting (merge sort, quick sort)
- **O(n²)**: Quadratic - nested loops
- **O(n³)**: Cubic - triple nested loops
- **O(2ⁿ)**: Exponential - generating all subsets
- **O(n!)**: Factorial - generating all permutations

#### Space Complexities
- **O(1)**: Constant space - using fixed number of variables
- **O(n)**: Linear space - storing array, hash table
- **O(n²)**: Quadratic space - 2D array, adjacency matrix

#### Common Operations
- **Array access**: O(1)
- **Array search**: O(n)
- **Array insert/delete**: O(n)
- **Hash table operations**: O(1) average, O(n) worst
- **Binary search**: O(log n)
- **Tree traversal**: O(n)
- **Graph traversal**: O(V + E)
- **Sorting**: O(n log n) for comparison-based sorts

---

## Quick Reference Tips

1. **When to use Two Pointers**: Sorted arrays, palindromes, removing duplicates
2. **When to use Sliding Window**: Subarrays, substrings with constraints
3. **When to use Binary Search**: Sorted arrays, searching in sorted space
4. **When to use DFS**: Tree/graph traversal, backtracking, path finding
5. **When to use BFS**: Level-order traversal, shortest path (unweighted)
6. **When to use DP**: Overlapping subproblems, optimal substructure
7. **When to use Greedy**: Local optimal leads to global optimal
8. **When to use Backtracking**: Generate all solutions, constraint satisfaction
9. **When to use Heap**: K largest/smallest, priority-based processing
10. **When to use Trie**: String prefix matching, dictionary problems

---

*Last updated: 2024*

