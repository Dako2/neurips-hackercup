
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
"""

ANALYZER_PROMPT = """{problem_statement} ##Output: What's the range of main variable n in this problem? What's the level of time complexity the code should hit for? Provide the Minimum time complexity for this problem according to the reference.\n """
