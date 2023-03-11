from collections import defaultdict
def get_shortest_unique_substring(arr, str):
  if not str: return ""
  counts = defaultdict(int)  
  for ch in arr:
    counts[ch] += 1
  have =  0
  need = len(arr)
  
  left = 0
  right = 0
  smallest = float("inf")
  start = None
  freq = {}
  while right < len(str):
    ch = str[right]
    
    freq[ch] = freq.get(ch, 0) + 1
    
    if ch in counts and freq[ch] == counts[ch]:
      have += 1
      
      
    while have == need:
      if smallest > right - left + 1:
        smallest = right - left + 1
        start = left
      ch = str[left]
      freq[ch] -= 1
      
      if ch in counts and freq[ch] < counts[ch]:
        have -= 1
        
      left += 1
      
    right += 1
  print(start, smallest)
  if smallest == float("inf"): return ""
    
  return str[start: start+smallest]

arr = ["A"]
str = "A"
# arr = ['x','y','z']
# str = "xyyzyzyx"
print(get_shortest_unique_substring(arr, str))
    
    