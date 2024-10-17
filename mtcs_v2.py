import shutil
from pathlib import Path
import logging
import math
import random
import numpy as np
from lib.utils import (
    create_logger,
    load_problem_from_folder,
    verify_code_syntax,
    extract_text,
    maybe_remove_backticks,
    save_to_disk,
)
from lib.llms import LLM
from lib.prompts import (
    CODER_INSTRUCTIONS,
)
from solution import SolutionManager, Solution

class Node:
    def __init__(self, state, parent=None, evaluation=None):
        self.state = state  # The current solution or state
        self.parent = parent  # Reference to the parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.score = 0  # The evaluation score of this node
        self.evaluation = evaluation  # Store the evaluation feedback
        self.reward_samples = []
        self.Q = 0  # Quality score
    
    def add_child(self, child_state):
        """Add a child node to this node."""
        child = Node(state=child_state, parent=self)
        self.children.append(child)
        return child
    
    def update_score(self, value):
        """Update score and visits, typical for backpropagation in MCTS."""
        self.score += value
        self.visits += 1
        if self.parent:
            self.parent.update_score(value)  # Propagate the score up the tree

    def add_reward(self, reward):
        """Update node quality based on the reward."""
        self.reward_samples.append(reward)
        avg_reward = np.mean(self.reward_samples)
        min_reward = np.min(self.reward_samples)
        self.Q = (avg_reward + min_reward) / 2  # Average of worst and avg outcome


class MCTS:
    def __init__(self, llm_model, problem, logger=None):
        self.problem = problem
        self.root = Node(state=None)  # The initial root node
        if not logger:
            self.logger = create_logger(f'logs/MCTS_{problem.problem_name}_{llm_model}.log', f'{problem.problem_name}_{llm_model}')
        self.llm = LLM(model_name=llm_model)
        self.fast_llm = LLM(model_name='gpt4')
        self.sm = SolutionManager()
        self.model_name = llm_model
    
    def ucb1(self, node, exploration_weight=1.4):
        """Calculate the UCB1 score for a node."""
        if node.visits == 0:
            return float('inf')  # Explore unvisited nodes first
        return node.score / node.visits + exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
    
    def select(self, node):
        """Select the best child node to explore based on UCB1."""
        return max(node.children, key=lambda n: self.ucb1(n))
        
    def expand(self, node, problem):
        """Expand the node by generating child nodes, incorporating evaluation feedback."""
        messages = self.build_prompt_with_feedback(node, problem)
         
        self.logger.info(f"\n\n***************: Competitor is running...***************\n\n")
        self.logger.info(f"Input: {messages}")           
        n = 2  # Number of child nodes to generate
        response = self.llm.mcts_openai_messages(messages, temperature=1, n=n)
        
        # Generate and add child nodes based on the response
        for i in range(n):
            out = response.choices[i].message.content.strip()
            self.logger.info(f"Output[{i}]: {out}")
            child_node = node.add_child(out)
            child_node.parent = node  # Ensure the parent is set
            
    def build_prompt_with_feedback(self, node, problem):
        """Construct the prompt including previous solutions and their evaluations."""
        prompt = f"""Your goal is to provide the TRULY correct and NO-TIMEOUT solution.
        ## Problem:
        {problem.problem_description}
        """
        current_node = node
        conversation_history = []
        while current_node.parent is not None:
            summarized_state = self.summarize_evaluation(current_node.state)
            conversation_history.append({
                'role': 'assistant',
                'content': summarized_state + current_node.evaluation
            })
            current_node = current_node.parent
        
        conversation_history = conversation_history[::-1]
        messages = [{'role': 'user', 'content': prompt}]
        messages.extend(conversation_history)
         
        return messages
    
    def summarize_evaluation(self, evaluation_text):
        """Summarize the evaluation text to reduce length."""
        max_length = 200
        if len(evaluation_text) > max_length:
            return evaluation_text[-max_length:] + '...'
        else:
            return evaluation_text

    def simulate(self, node, problem):
        """Run the simulation (AI solution generation and evaluation)."""
        out = node.state  # Get the current solution
        code = self.worker(out)  # Use the worker to process the solution

        self.logger.info(f"Simulating: Output is {out}")
        s = Solution(code, problem.problem_name, problem.sample_input_path,
                    problem.sample_output_path, problem.full_input_path, self.model_name)
        testreport, fullreport = s.eval()
        self.sm.add_solution(s)

        node.evaluation = f"<sample_test>{testreport}</sample_test>\n<full_test>{fullreport}</full_test>"
        score = testreport.success_rate_number
        if fullreport and fullreport.status == "timeout":
            score -= 0.2  # Penalty for timeout
        
        self.backpropagate(node, score)

        self.logger.info(f"Solution evaluated. Score: {score}")
        return s.to_submit_signal

    def backpropagate(self, node, reward):
        """Backpropagate the result up the tree."""
        while node is not None:
            node.add_reward(reward)
            node = node.parent

    def self_refine(self, node):
        """Refine the solution using LLM feedback."""
        messages = self.build_prompt_with_feedback(node, self.problem)
        self.logger.info(f"Refining the solution for node {node.state}")
        response = self.llm.mcts_openai_messages(messages)
        refined_solution = response.choices[0].message.content
        refined_node = node.add_child(refined_solution)
        self.simulate(refined_node, self.problem)
        return refined_node

    def zero_shot_initialization(self):
        """Generate an initial solution for the root node."""
        prompt = f"Problem: {self.problem.problem_description}\nGenerate a solution."
        response = self.llm.mcts_openai_messages([{'role': 'user', 'content': prompt}])
        initial_solution = response.choices[0].message.content
        self.root = Node(state=initial_solution)
    
    def mcts_trial(self, problem, max_steps=10):
        """Run MCTS trial to find a solution."""
        self.zero_shot_initialization()
        step = 0
        current_node = self.root

        while step < max_steps:
            while current_node.children:
                current_node = self.select(current_node)

            self.expand(current_node, problem)

            for child in current_node.children:
                to_submit_signal = self.simulate(child, problem)

                if to_submit_signal:
                    self.logger.info("Problem solved, ready for submission.")
                    return child  # Return the successful solution

            step += 1

        self.logger.info("Max steps reached without finding a solution.")
        return None

    def worker(self, assistant_output):
        """Processes assistant output to extract and verify the source code."""
        messages = [{'role': 'user', 'content': CODER_INSTRUCTIONS.format(code=assistant_output)}]
        out = self.fast_llm.run_messages(messages=messages)
        code = extract_text(out, '<source_code>')
        code = maybe_remove_backticks(code)
    
        if verify_code_syntax(code):
            return code
        else:
            return ""
 
if __name__ == '__main__':
    logger = create_logger(f'logs/trainer.log', 'trainer')
    
    from lib.utils import load_problem_training, list_problem_names
    problem_directory = "/mnt/d/AIHackercup/dataset/2023/round2"
    problem_names = list_problem_names(problem_directory, "2023")
     
    for problem_name in problem_names[2:3]:
        problem = load_problem_training(problem_name, Path(problem_directory))
        solution_guidelines = problem.solution
 
        model_name = 'gpt3.5'
        mcts = MCTS(model_name, problem)
        solution_node = mcts.mcts_trial(problem, max_steps=10)
        print(mcts.sm.solution_manager)
        mcts.sm.to_submit('to_submit/')
