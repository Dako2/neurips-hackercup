
SELECT_LANGUAGE = 'cpp'

# Using language coding
EXTRACT_CODE_PROMPT = """Extract the code from the response and update it as neccessary, ensuring it's an executable {selected_language} program.
- ***code only!*** Remove any surrounding ```{selected_language}` tags or explanations. 
- For C++:
- Use only standard libraries; **do not use or include `bits/stdc++.h`**.
- Remember declare or define identifier (variable, function, etc.)
- **Include necessary standard headers explicitly at the beginning of the code**. Include other headers as needed, such as <cmath>, <limits>, <functional>. For example:
#include <iostream>
#include <vector>
#include <algorithm>
- For Python:
- Include an `if __name__ == "__main__":` block.
- DO NOT USE threading

Current response containing code:
{output}
The original problem statement:
{problem}
"""

ANALYZER_PROMPT = """{problem_statement} ##Output: What's the range of main variable n in this problem? What's the level of time complexity the code should hit for? Provide the Minimum time complexity for this problem according to the reference.\n """


TIME_COMPLEXITY_ANALYZER_PROMPT= """## Problem: {problem_statement}

By looking at the constraints of a problem, we can often "guess" the solution.
Common time complexities
Let n be the main variable in the problem.
If n ≤ 12, the time complexity can be O(n!).
If n ≤ 25, the time complexity can be O(2^n).
If n ≤ 100, the time complexity can be O(n^4).
If n ≤ 500, the time complexity can be O(n^3).
If n ≤ 10^4, the time complexity can be O(n^2).
If n ≤ 10^6, the time complexity can be O(n log n).
If n ≤ 10^8, the time complexity can be O(n).
If n > 10^8, the time complexity can be O(log n), O(sqrt(n)) or O(1).
Examples of each common time complexity

O(n!) [Factorial time]: Permutations of 1 ... n
O(2^n) [Exponential time]: Exhaust all subsets of an array of size n
O(n^3) [Cubic time]: Exhaust all triangles with side length less than n
O(n^2) [Quadratic time]: Slow comparison-based sorting (eg. Bubble Sort, Insertion Sort, Selection Sort)
O(n log n) [Linearithmic time]: Fast comparison-based sorting (eg. Merge Sort)
O(n) [Linear time]: Linear Search (Finding maximum/minimum element in a 1D array), Counting Sort
O(log n) [Logarithmic time]: Binary Search, finding GCD (Greatest Common Divisor) using Euclidean Algorithm
O(sqrt(n))
O(1) [Constant time]: Calculation (eg. Solving linear equations in one unknown)
Explanations based on Codeforces problems

Calculate the time complexity requirements based on the above guidelines and problem statement in 20 words.
"""

STACK_FLOW_COMMAND=''
#'-Wl,-stack_size,0x20000000'
#Linux: ulimit -s 