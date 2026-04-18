---
layout: post
title: "Shortest Unique Substring (Sliding Window)"
category: others
topic: algorithms
date: 2023-01-01
description: "Find the shortest substring of a string that contains all characters from a given array."
---

Given an array of unique characters and a string, find the shortest substring that contains all characters from the array.

## Approach

Use a sliding window with two pointers. Expand `right` to include required characters; shrink `left` once all are covered.

**Time complexity:** $O(n)$ — each character is visited at most twice.  
**Space complexity:** $O(k)$ where $k$ is the size of the character array.

## Code

```python
from collections import defaultdict

def get_shortest_unique_substring(arr, s):
    if not s:
        return ""

    counts = defaultdict(int)
    for ch in arr:
        counts[ch] += 1

    have, need = 0, len(arr)
    left = 0
    smallest = float("inf")
    start = None
    freq = {}

    for right in range(len(s)):
        ch = s[right]
        freq[ch] = freq.get(ch, 0) + 1

        if ch in counts and freq[ch] == counts[ch]:
            have += 1

        while have == need:
            if right - left + 1 < smallest:
                smallest = right - left + 1
                start = left

            left_ch = s[left]
            freq[left_ch] -= 1
            if left_ch in counts and freq[left_ch] < counts[left_ch]:
                have -= 1
            left += 1

    return "" if smallest == float("inf") else s[start:start + smallest]
```

## Example

```python
print(get_shortest_unique_substring(['x', 'y', 'z'], "xyyzyzyx"))
# Output: "zyx"
```
