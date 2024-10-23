import random
import math
import logging
import urllib.request
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from collections import defaultdict
import re 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def verify_code_syntax2(code):
    """
    Verifies the syntax of the provided Python code.
    Returns (True, code) if syntax is correct, (False, error_message) otherwise.
    """
    try:
        compile(code, '<string>', 'exec')
        return True, code
    except SyntaxError as e:
        return False, str(e)

def extract_text(input_string, format):
    # Use a regex pattern to extract text between <prompt> and </prompt>
    #match = re.search(f'{format}(.*?){format.replace('<','</')}', input_string, re.DOTALL)
    match = re.search(f'{format}(.*?){format.replace("<", "</")}', input_string, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return input_string

def extract_python_code(text):
    # Use regex to find the content between ```python and ```
    pattern = r"```python(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)

    # Return the extracted code blocks
    return code_blocks[0]
    
def maybe_remove_backticks(solution: str) -> str:
    "Remove backticks from the solution"
    solution = solution.strip()
    solution = re.sub(r'^```python\s*', '', solution)
    solution = re.sub(r'\s*```$', '', solution)
    return solution

def save_to_disk(content: str, path: Path,):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(content)
    with path.open("w", encoding ="utf-8") as f:
        f.write(content)

# Function to query the LLM (your 'query_model' function)
def call_llm(prompt, model="llama3.1", seed=None, url="http://localhost:11434/api/chat"):
    """
    Calls the LLM with the given prompt and returns the response.
    """
    if seed is None:
        seed = random.randint(1, 1000)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        "options": {
            "seed": seed,
            "temperature": 1,
            "num_ctx": 2048  # Ensure consistent output
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    try:
        with urllib.request.urlopen(request) as response:
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_data += response_json["message"]["content"]
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return ""

    #logger.info(f"LLM Response: {response_data}")
    return response_data.strip()

# Function to execute code safely (adapted from your 'Solution' class)
def execute_code(code, test_case_input, test_case_output):
    """
    Executes the provided code with the test case input and compares the output.
    Returns True if the output matches the expected output, False otherwise.
    """
    # Write the code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file.write(code)
        tmp_file_path = tmp_file.name

    # Run the code using subprocess
    try:
        # Prepare the input data
        input_data = '\n'.join(test_case_input)

        # Execute the code
        result = subprocess.run(
            [sys.executable, tmp_file_path],
            input=input_data,
            text=True,
            capture_output=True,
            timeout=10  # Set a timeout to prevent infinite loops
        )

        # Get the output and compare with expected output
        output = result.stdout.strip()
        expected_output = '\n'.join(test_case_output).strip()

        # Clean up the temporary file
        Path(tmp_file_path).unlink()

        return output == expected_output

    except subprocess.TimeoutExpired:
        logger.error("Code execution timed out.")
        Path(tmp_file_path).unlink()
        return False
    except Exception as e:
        logger.error(f"Error during code execution: {e}")
        Path(tmp_file_path).unlink()
        return False

class Node:
    def __init__(self, code_state, parent=None, action=None):
        self.code_state = code_state  # The current code as a string
        self.parent = parent
        self.children = []
        self.visits = 0
        self.Q = 0.0
        self.N_sa = defaultdict(int)
        self.Q_sa = defaultdict(float)
        self.action = action  # Modification that led to this state
        self.untried_actions = self.get_possible_actions()
        self.tried_actions = set()

    def get_possible_actions(self):
        """
        Defines possible actions (code modifications) for this node.
        """
        return [
            "Improve code efficiency",
            "Fix syntax errors",
            "Handle edge cases",
            "Optimize loops",
            "Correct logical errors",
            "Add comments for clarity",
            "Refactor code structure"
        ]

    def is_terminal(self):
        # Define terminal condition, e.g., code passes all test cases
        return self.Q == 1.0  # If Q-value is 1.0, all test cases passed

class SR_MCTS_LLM:
    def __init__(self, initial_code, test_cases, max_nodes, exploration_constant=1.4, alpha=0.5, gamma=0.9):
        self.root = Node(initial_code)
        self.test_cases = test_cases  # List of (input, expected_output)
        self.max_nodes = max_nodes
        self.c = exploration_constant
        self.alpha = alpha
        self.gamma = gamma
        self.tree_size = 1
        self.iteration = 0

    def search(self):
        while self.tree_size < self.max_nodes:
            self.iteration += 1
            logger.info(f"\nIteration {self.iteration}: Tree Size = {self.tree_size}")
            # Selection Phase
            node = self.select(self.root)
            if node is None:
                logger.info("No node selected for expansion (all nodes pruned or terminal).")
                break
            logger.info(f"Selected node for expansion: Node ID {id(node)}")
            # Expansion Phase
            child_node = self.expand(node)
            logger.info(f"Expanded node: Node ID {id(child_node)}")
            # Evaluation Phase
            q_value = self.evaluate(child_node)
            logger.info(f"Evaluation of node ID {id(child_node)}: Q-value = {q_value:.4f}")
            # Backpropagation Phase
            self.backpropagate(child_node, q_value)
            # Check for terminal solution
            if child_node.is_terminal():
                logger.info("Terminal solution found.")
                break
        # Return the best solution found
        return self.get_best_solution()

    def select(self, node):
        while True:
            if node.is_terminal():
                return node
            if len(node.untried_actions) > 0:
                return node
            else:
                node = self.best_uct(node)
                if node is None:
                    return None

    def best_uct(self, node):
        best_value = -float('inf')
        best_nodes = []
        for child in node.children:
            action = child.action
            q_sa = node.Q_sa[action]
            n_sa = node.N_sa[action]
            n_s = node.visits
            if n_sa == 0:
                uct_value = float('inf')
            else:
                uct_value = q_sa + self.c * math.sqrt(math.log(n_s) / n_sa)
            if uct_value > best_value:
                best_value = uct_value
                best_nodes = [child]
            elif uct_value == best_value:
                best_nodes.append(child)
        if not best_nodes:
            return None
        selected_node = random.choice(best_nodes)
        return selected_node

    def expand(self, node):
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        node.tried_actions.add(action)
        # Use the action as the critique
        critique = self.critique(node.code_state, action)
        new_code = self.rewrite(node.code_state, critique)
        child_node = Node(new_code, parent=node, action=action)
        node.children.append(child_node)
        self.tree_size += 1
        return child_node

    def critique(self, code_state, action): #manager role
        # For simplicity, we use the action as the critique
        critique = action
        prompt = f"The following code needs improvement:\n{code_state}\nCritique: {critique}\nPlease provide a corrected and improved version of the code."
        new_code = call_llm(prompt)
        return critique

    def rewrite(self, code_state, critique): #worker role
        # Use LLM to generate improved code based on the critique
        prompt = f"The following code needs improvement:\n{code_state}\nCritique: {critique}\nPlease provide a corrected and improved version of the code."
        new_code = call_llm(prompt)
        return new_code

    def evaluate(self, node):
        # Verify code syntax
        
        code = extract_text(node.code_state, '<source_code>')
        code = extract_python_code(code)
        code = maybe_remove_backticks(code)
        code_exec_flag, verified_code_or_error = verify_code_syntax2(code)
        
        logger.info(f"Code:''' {code}'''\n")
        if not code_exec_flag:
            logger.error(f"Code syntax error: {verified_code_or_error}")
            qg = 0.0
        else:
            # Run the code against test cases and compute a score
            pass_count = 0
            for test_input_lines, test_output_lines in self.test_cases:
                if execute_code(node.code_state, test_input_lines, test_output_lines):
                    pass_count += 1
            qg = pass_count / len(self.test_cases)
        ql = self.local_value(node)
        q_value = self.alpha * qg + (1 - self.alpha) * ql
        node.Q = q_value
        return q_value

    def local_value(self, node):
        # Use the average Q value of siblings
        if node.parent is None:
            return 0.0
        sibling_values = [sibling.Q for sibling in node.parent.children if sibling is not node]
        if sibling_values:
            ql = sum(sibling_values) / len(sibling_values)
        else:
            ql = node.parent.Q
        return ql

    def backpropagate(self, node, q_value):
        while node is not None:
            node.visits += 1
            node.Q = (1 - self.gamma) * node.Q + self.gamma * q_value
            if node.parent is not None:
                action = node.action
                node.parent.N_sa[action] += 1
                node.parent.Q_sa[action] = (1 - self.gamma) * node.parent.Q_sa[action] + self.gamma * q_value
            node = node.parent

    def get_best_solution(self):
        # Return the code with the highest Q value found
        best_node = self.root
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            current_node = nodes_to_visit.pop()
            if current_node.Q > best_node.Q:
                best_node = current_node
            nodes_to_visit.extend(current_node.children)
        if best_node:
            logger.info(f"Best solution found with Q-value {best_node.Q:.4f}")
            return best_node.code_state
        else:
            logger.info("No valid solution found.")
            return None

# Sample competitive coding problem: Two Sum
# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    
from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training, load_problem_v2024
from pathlib import Path

"""
problem_directory = "dataset/2023/round2"
problem_names = list_problem_names(problem_directory, "2023")
problem_list = []
for problem_name in problem_names:
    problem_list.append(load_problem_training(problem_name, Path(problem_directory)))
"""

problem_directory = "contestData"
problem_names = list_problem_names(problem_directory, "2024")
problem_list = []
for problem_name in problem_names:
    problem_list.append(load_problem_v2024(problem_name, Path(problem_directory)))

problem = problem_list[1]
# Initial code (could be empty or a simple template)
initial_code = problem.problem_description

test_cases = problem.sample_input

# Instantiate the SR_MCTS_LLM algorithm
sr_mcts_llm = SR_MCTS_LLM(initial_code, test_cases, max_nodes=10)

# Perform the search
best_solution = sr_mcts_llm.search()

# Output the best solution found
print("\nBest Solution Found:")
print(best_solution)
