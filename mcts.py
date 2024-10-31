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
from solution import Solution, SolutionManager
from lib.prompts import CODER_INSTRUCTIONS
from lib.llms import LLM
from lib.utils import (
    create_logger,
    load_problem_from_folder,
    verify_code_syntax,
    extract_text,
    maybe_remove_backticks,
    save_to_disk,
)
import json 
from sentence_transformers import SentenceTransformer, util

# Configure logging
logger = create_logger(f'logs/sr_MCTS_.log', f'gpt4')
#logging.basicConfig(level=logging.INFO, format='%(message)s')
#logger = logging.getLogger()

# Load the model globally (only once)
model = SentenceTransformer('all-MiniLM-L6-v2')

from config import SELECT_LANGUAGE, EXTRACT_CODE_PROMPT, ANALYZER_PROMPT


def check_similarity(new_code, previous_codes, threshold=0.9):
    new_embedding = model.encode(new_code, convert_to_tensor=True)
    for prev_code in previous_codes:
        prev_embedding = model.encode(prev_code, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(new_embedding, prev_embedding).item()
        if similarity > threshold:
            return True  # Prune if similar
    return False

class Node:
    def __init__(self, state, code, depth=0, parent=None, action=None, evaluation=None, prompt=None):
        self.state = state  # The current code as a string
        self.parent = parent
        self.children = []
        self.visits = 0
        self.Q = 0.0
        self.score = 0
        self.N_sa = defaultdict(int)
        self.Q_sa = defaultdict(float)
        self.action = action  # Modification that led to this state
        self.evaluation = evaluation
        self.code = code
        self.prompt = prompt
        self.tried_actions = set()
        self.depth = depth  # Track the depth of the node
        if self.depth == 0:
            #self.untried_actions = get_possible_actions(plans_xml)
            self.untried_actions = self.get_possible_actions()
        elif self.depth > 0 and self.depth < 2:
            self.untried_actions = self.get_possible_actions()    
        else:
            self.untried_actions = self.get_possible_actions()       
        self.uniqueness_score=1
        self.solution = None
        self.terminal = False
        
    def get_possible_actions(self):
        """
        Defines possible actions (code modifications) for this node.
        """
        return [
            "Refine the method to improve the correction and time complexity",
            #"Adjust logic for fewer loops.",
        ]

    def add_child(self, state, code, parent, action=None, prompt=None):
        """Add a child node to this node."""
        child = Node(state=state, code=code, depth=self.depth+1, parent=self, action=action, prompt=prompt)
        self.children.append(child)
        return child

    def is_terminal(self):
        # Define terminal condition, e.g., code passes all test cases
        return self.terminal

def print_tree(node: Node, prefix: str = "", current_node: Node = None):
    if node is None:
        return

    # Determine connector symbol for tree structure
    connector = "└─" if node.parent and node.parent.children[-1] == node else "├─"
     
    # Highlight the current node with a marker (e.g., "⇨")
    marker = "⇨ ⇨ ⇨ ⇨ here!" if node == current_node else " "

    # Print node details, including the marker for the current node
    print(f"{prefix}{connector} Depth={node.depth}, State={id(node)}, Score = {node.score:.2f}, Uniqueness={node.uniqueness_score:.2f}, Q={node.Q:.2f}, Visits={node.visits} {marker}")

    # Update prefix for children based on this node's position
    new_prefix = prefix + ("   " if connector == "└─" else "│  ")

    # Recursively print each child node
    for child in node.children:
        print_tree(child, new_prefix, current_node)

class SR_MCTS_LLM:
    def __init__(self, problem, max_nodes, exploration_constant=1.4, alpha=0.5, gamma=0.9):
        self.root = Node(problem.problem_description, "")
        self.problem = problem  # List of (input, expected_output)
        self.max_nodes = max_nodes
        self.c = exploration_constant
        self.alpha = alpha
        self.gamma = gamma
        self.tree_size = 1
        self.iteration = 0
        self.sm = SolutionManager()
        self.fast_llm = LLM(model_name='gpt4')
        self.model_name = "gpt4"
        self.strong_llm = LLM(model_name=self.model_name)

    def search(self):
        previous_solutions = set()  # Store previously generated solutions

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
            child_node, q_value = self.expand(node)
            print_tree(self.root, current_node=child_node)
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
        # Use the action as the critique
        action = self.get_action(node)
        
        critique, prompt = self.manager(node, action)
        new_code = self.worker(node, critique)
        
        child_node = node.add_child(state=critique, code=new_code, parent=node, action=action, prompt=prompt)
        self.tree_size += 1
        logger.info(f"Expanded node: Node ID {id(child_node)}")
        
        # Evaluation Phase
        q_value = self.evaluate(child_node)
        logger.info(f"Evaluation of node ID {id(child_node)}: Q-value = {q_value:.4f}")
        
        return child_node, q_value

    def get_action(self, node, option=True):
        #based on the previous actions of all the child nodes under this parent node, generate a new action
        #there are two approaches: 1. defined list of actions; 2. undefined LLM generated action
        if option:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            node.tried_actions.add(action)
        else:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            node.tried_actions.add(action)
        return action 
    
    def strategist(self, node):
        pass

    def manager(self, node, action):
        # For simplicity, we use the action as the critique
        messages = self.build_prompt_with_feedback(node)
        messages.extend([{'role': 'user', 'content': action}])
        logger.debug(f"\n@_@ - Messages:{messages}")
        response = self.strong_llm.run_messages(messages)
        logger.debug(f"\n@_@ - response:{response}")
        return response, json.dumps(messages)
    
    def worker(self, node, critique): #worker role
        # Use LLM to generate improved code based on the critique
        new_code = self.coder(critique)
        node.code = new_code
        logger.debug(f"\n@_@ - New Code:{new_code}")
        return new_code

    def coder(self, critique): #implement the code; fixed context length simple response
        """Processes assistant output to extract and verify the source code."""
        messages = [{'role': 'user', 
                     'content': EXTRACT_CODE_PROMPT.format(
                                output=critique,
                                selected_language=SELECT_LANGUAGE)
                     }]
        out = self.fast_llm.run_messages(messages=messages)
        
        code = extract_text(out, '<source_code>')
        code = maybe_remove_backticks(code)
        return code

    def build_prompt_with_feedback(self, node):
        """Construct the prompt including previous solutions and their evaluations."""
        prompt = f"""Provide TRULY correct and NO-TIMEOUT solution in {SELECT_LANGUAGE}. Problem: {self.problem.problem_description}"""
        current_node = node
        conversation_history = []
        while current_node.parent is not None:
            conversation_history.append(
            {
                'role': 'user',
                'content': current_node.evaluation,
            },)
            summarized_state = self.summarize_state(current_node.state)
            conversation_history.append({
                'role': 'assistant',
                'content': summarized_state,
            })
            current_node = current_node.parent
        conversation_history = conversation_history[::-1]
        messages = [{'role': 'user', 'content': prompt}]
        messages.extend(conversation_history)
        return messages
    
    def summarize_state(self, evaluation_text):
        """Summarize the evaluation text to reduce length."""
        max_length = 200
        if len(evaluation_text) > max_length:
            return evaluation_text[:max_length] + '...'+ evaluation_text[-max_length:] 
        else:
            return evaluation_text
 
    def heuristic_score(self, uniqueness_score, correct, full_status):
        if full_status=='timeout':
            efficiency = -10
        elif full_status=='pending':
            efficiency = 0
        else:
            efficiency = 1
 
        return correct + 0.5 * efficiency + 0.5 * uniqueness_score

    def evaluate(self, node):
        # Verify code syntax
        new_code = node.code
        """
        try:
            uniqueness_score = 1 - max(
                util.pytorch_cos_sim(model.encode(new_code, convert_to_tensor=True), model.encode(code, convert_to_tensor=True)).item()
                for code in [n.code for n in self.sm.solution_manager.code.values]
            )
        except:
            uniqueness_score = 1
        """
        uniqueness_score = 1
        
        s = Solution(new_code, self.problem, self.model_name, "mcts", lang=SELECT_LANGUAGE) #py, cpp

        testreport, fullreport = s.eval(logger)
        if testreport.status == "passed" and fullreport.status == "complete":
            node.terminal = True

        node.evaluation = f"{self.problem.problem_name}:\n - Sample test: {testreport.message}\n - Full test: {fullreport.as_xml}"
        node.uniqueness_score = uniqueness_score
        logger.info(f"\n@_@ - Node: {node.evaluation}")
        
        qg = self.heuristic_score(uniqueness_score, testreport.success_rate_number, fullreport.status)  # Score based on success rate for failed cases        
        ql = self.local_value(node) 
        q_value = self.alpha * ql + (1 - self.alpha) * qg
        
        node.Q = q_value
        node.score = testreport.success_rate_number
        node.uniqueness_score = uniqueness_score
        node.solution = s

        #some logisitics
        s.solver = "sr_mcts"
        s.model_capability = "gpt4"
        s.score = qg
        s.q = q_value
        s.prompt =  node.prompt
        self.sm.add_solution(s)

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
            return best_node.state
        else:
            logger.info("No valid solution found.")
            return None

def convert_xml_to_list(plans_xml):
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(plans_xml.strip())
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        print(f"XML content: {plans_xml}")
    # Create a list of solutions
    plan_list = []
    for solution in root.findall('solution'):
        method = solution.find('method').text
        description = solution.find('description').text
        complexity = solution.find('complexity').text
        steps = [step.text for step in solution.find('steps').findall('step')]
        
        plan_list.append(json.dumps({
            'method': method,
            'description': description,
            'complexity': complexity,
            'steps': steps
        }))        
    return plan_list

def get_possible_actions(plans_xml):
    """
    Defines possible actions (code modifications) for this node.
    """
    plans_xml = maybe_remove_backticks(plans_xml, 'xml')
    plan_list = convert_xml_to_list(plans_xml)
    return plan_list

if __name__ == '__main__':
        
    from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training, load_problem_v2024
    from pathlib import Path

    """
    problem_directory = "dataset/2023/round2"
    problem_names = list_problem_names(problem_directory, "2023")
    problem_list = []
    for problem_name in problem_names:
        problem_list.append(load_problem_training(problem_name, Path(problem_directory)))
    """

    import xml.etree.ElementTree as ET

    plans_xml = '''

    '''

    problem_directory = "dataset/2024/Round2"
    problem_names = list_problem_names(problem_directory, "2024")
    problem_list = []
    for problem_name in problem_names:
        problem_list.append(load_problem_v2024(problem_name, Path(problem_directory)))
    problem = problem_list[1]

    # Instantiate the SR_MCTS_LLM algorithm
    sr_mcts_llm = SR_MCTS_LLM(problem, max_nodes=30)
    # Perform the search
    best_solution = sr_mcts_llm.search()

    # Output the best solution found
    print("\nBest Solution Found:")
    print(best_solution)

    print(sr_mcts_llm.sm.solution_manager)

    sr_mcts_llm.sm.save(file_path=f"sm_sr_mcts_cpp_{problem.problem_name}")
    print_tree(sr_mcts_llm.root, "", sr_mcts_llm.root)

