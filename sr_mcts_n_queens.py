import random
import math
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Mock LLM function (Replace with actual LLM API calls)
def call_llm(prompt):
    # Simulate LLM output based on the prompt
    # In practice, use an API call to OpenAI's GPT models
    response = """
def two_sum(nums, target):
    num_to_index = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return [num_to_index[complement], i]
        num_to_index[num] = i
    return []
"""
    return response.strip()

# Mock code execution function (Replace with safe execution environment)
def execute_code(code, test_case):
    try:
        # Prepare the environment
        local_env = {}
        exec(code, {}, local_env)
        func = local_env.get('two_sum')
        if func is None:
            return False
        # Run the test case
        nums, target = test_case
        result = func(nums, target)
        # Expected results for the sample test cases
        expected_results = {
            (2, 7, 11, 15, 9): [0, 1],
            (3, 2, 4, 6): [1, 2],
            (3, 3, 6): [0, 1]
        }
        expected = expected_results.get(tuple(nums + [target]), [])
        return sorted(result) == sorted(expected)
    except Exception as e:
        logger.debug(f"Execution error: {e}")
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

    def critique(self, code_state, action):
        # For simplicity, we use the action as the critique
        critique = action
        return critique

    def rewrite(self, code_state, critique):
        # Use LLM to generate improved code based on the critique
        prompt = f"The following code needs improvement:\n{code_state}\nCritique: {critique}\nPlease provide a corrected and improved version of the code."
        new_code = call_llm(prompt)
        return new_code

    def evaluate(self, node):
        # Run the code against test cases and compute a score
        pass_count = 0
        for test_input, target in self.test_cases:
            if execute_code(node.code_state, (test_input, target)):
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

# Test cases
test_cases = [
    ([2, 7, 11, 15], 9),    # Expected output: [0, 1]
    ([3, 2, 4], 6),         # Expected output: [1, 2]
    ([3, 3], 6)             # Expected output: [0, 1]
]

# Initial code (could be empty or a simple template)
initial_code = """
def two_sum(nums, target):
    # Your code here
    pass
"""

# Instantiate the SR_MCTS_LLM algorithm
sr_mcts_llm = SR_MCTS_LLM(initial_code, test_cases, max_nodes=10)

# Perform the search
best_solution = sr_mcts_llm.search()

# Output the best solution found
print("\nBest Solution Found:")
print(best_solution)
