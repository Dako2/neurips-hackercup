import shutil
from pathlib import Path
import logging
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
    REFLECTION_INSTRUCTIONS_USER,
    REFLECTION_INSTRUCTIONS_SYSTEM,
    initial_advisor_prompt,
    extract_prompt,
    CODER_INSTRUCTIONS,
    manager_prompt,
    prompt_rephrase_problem,
    SOLVER_INSTRUCTIONS,
    OUTPUT_FORMAT_CLASSIFIER,
)
from solution import SolutionManager, Solution
import math
import random

class Node:
    def __init__(self, state, parent=None, evaluation=None):
        self.state = state  # The current solution or state
        self.parent = parent  # Reference to the parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.score = 0  # The evaluation score of this node
        self.evaluation = evaluation  # Store the evaluation feedback
    
    def add_child(self, child_state):
        """Add a child node to this node."""
        child = Node(state=child_state, parent=self)
        self.children.append(child)
        return child
    
    def update_score(self, value):
        """Update score and visits, typical for backpropagation in MCTS."""
        self.score += value
        self.visits += 1
        
class MCTS:
    def __init__(self, llm_model, problem, logger=None):
        self.root = Node(state=None)  # The initial root node
        self.problem = problem
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
    
    def select(self, node): #select_best_child
        """Select the child with the highest score-to-visit ratio."""
        return max(node.children, key=lambda n: n.score / n.visits if n.visits > 0 else float('inf'))

    def select1(self, node):
        """Select the best child node to explore based on UCB1."""
        return max(node.children, key=lambda n: self.ucb1(n))
        
    def expand(self, node, problem):
        """Expand the node by generating child nodes, incorporating evaluation feedback."""
        # Build the prompt with previous solution and its evaluation
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
        # Start with the base problem description
        prompt = f"""Your goal is to provide the TRULY correct and NO-TIMEOUT solution.
    ## Problem:
    {problem.problem_description}
    """
        # Traverse up the tree to include previous solutions and evaluations
        current_node = node
        conversation_history = []
        while current_node.parent is not None:
            # Include the assistant's previous solution and its evaluation
            summarized_state = self.summarize_evaluation(current_node.state)
            conversation_history.append({
                'role': 'assistant',
                'content': summarized_state + current_node.evaluation
            })
            current_node = current_node.parent
        
        # Reverse the conversation history to chronological order
        conversation_history = conversation_history[::-1]
        
        # Construct the messages for the LLM
        messages = [{'role': 'user', 'content': prompt}]
        messages.extend(conversation_history)
         
        return messages
    
    def summarize_evaluation(self, evaluation_text): #TODO
        """Summarize the evaluation text to reduce length."""
        # For demonstration, we'll just truncate the text
        max_length = 200  # Adjust as needed
        if len(evaluation_text) > max_length:
            return evaluation_text[-max_length:] + '...' #TODO CHECK
        else:
            return evaluation_text

    def simulate(self, node, problem): #evaluation
        """Run the simulation (AI solution generation and evaluation)."""
        out = node.state  # Get the current solution
        code = self.worker(out)  # Use the worker to process the solution

        self.logger.info(f"Simulating: Output is {out}")
        s = Solution(code, problem.problem_name, problem.sample_input_path,
                    problem.sample_output_path, problem.full_input_path, self.model_name)
        testreport, fullreport = s.eval()
        self.sm.add_solution(s)

        # Store the evaluation feedback in the node
        node.evaluation = f"<sample_test>{testreport}</sample_test>\n<full_test>{fullreport}</full_test>"
        if fullreport:
            if fullreport.status not in ["timeout"]:
                score = -.2 #penality for timeout
        score = testreport.success_rate_number #self.evaluate_solution(testreport, fullreport)

        self.backpropagate(node, score)

        self.logger.info(f"Solution evaluated. Score: {score}")
        return s.to_submit_signal

    def backpropagate(self, node, reward):
        """Backpropagate the result up the tree."""
        current_node = node
        while current_node is not None:
            current_node.update_score(reward)
            current_node = current_node.parent

    def mcts_trial(self, problem, max_steps=10):#mcts_trial method orchestrates the MCTS process:
        step = 0
        current_node = self.root

        while step < max_steps:
            # Selection
            while current_node.children:
                current_node = self.select(current_node)

            # Expansion
            self.expand(current_node, problem)

            # Simulation
            for child in current_node.children:
                to_submit_signal = self.simulate(child, problem)

                # Check if solution is ready for submission
                if to_submit_signal:
                    self.logger.info("Problem solved, ready for submission.")
                    return child  # Return the successful solution

            step += 1

        self.logger.info("Max steps reached without finding a solution.")
        return None  # No solution found within the step limit


    def worker(self, assistant_output):
        """
        Processes assistant output to extract and verify the source code.
        """
        messages = [
            {
                'role': 'user',
                'content': CODER_INSTRUCTIONS.format(code=assistant_output)
            },
        ]
        out = self.fast_llm.run_messages(messages=messages)
        
        code = extract_text(out, '<source_code>')
        code = maybe_remove_backticks(code)
    
        if verify_code_syntax(code):
            #self.logger.info(f"Code syntax correct:\n{code}")
            return code
        else:
            return ""
            #raise ValueError("Source code is not compilable.")

def output_format_indicator(problem, logger):
    """
    Determines whether the problem requires an exact output format match or not.
    """
    llm = LLM('gpt4', logger=logger)
    messages = [
        {
            'role': 'user',
            'content': OUTPUT_FORMAT_CLASSIFIER.format(problem=problem.problem_description)
        },
    ]
    out = llm.run_messages(messages=messages, temperature=0.0)
    logger.info(f"{problem.problem_name} requires exact output format? {out}\n")
    return out

class Trainer:
    def __init__(self, model_name, problem, logger=None):
        self.problem = problem
        self.model_name = model_name
        if not logger:
            self.logger = create_logger(f'logs/{problem.problem_name}_{model_name}.log', '{problem.problem_name}_{model_name}')
        self.llm = LLM(model_name=model_name, logger=self.logger)
        self.fast_llm = LLM(model_name='gpt4', logger=self.logger)
        self.messages = []
        self.reflection_step = 0
        self.solution_list=[]
        self.sm = SolutionManager()

    @property
    def method_dict(self):
        return {
            "solve_problem_pro":self.solve_problem_pro,
            "reflection_pro":self.reflection_pro,
            "chain_of_thoughts":self.chain_of_thoughts,
            "interpreter":self.interpreter,
            "meta_learning":self.meta_learning,
        }
    def run(self, method):
        return self.method_dict[method]()
    
    def run(self, method):
        try:
            m = self.method_dict()[method]
            m()
        except:
            raise ValueError("method name error")
     

    def battle_ground(self):
        
        solution_list = []

        #'Subsonic Subway'
        #prompt1 = """You are Competitor #1. Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You NEVER agree with Competitor#2. You always provide the missing or misunderstood points by Competitor#2 and provide argument from the facts or derivations based on the input information. \n <root><problem_statement>{problem}</problem_statement></root>"""
        #prompt2 = """You are Competitor #2: Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You MAY learn from Competitor#1 but you always provide argument from the facts or derivations based on the input information. \n <root><problem_statement>{problem}</problem_statement></root>"""
        
        #'Prime Subtractorization'
        prompt1 = """You are Competitor #1: Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You NEVER agree with Competitor#2. You always provide argument from the facts or derivations based on the input information, and explicitly illustrate your NEW approach, fix and technique.\n <root><problem_statement>{problem}</problem_statement></root>"""
        prompt2 = """You are Competitor #2. Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You NEVER agree with Competitor#1. You always provide the overlook insights from the problem, provide NEW approach, provide fix and advanced techniques. \n <root><problem_statement>{problem}</problem_statement></root>"""
        
        #prompt1 = """You are Competitor #1: Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You NEVER agree with Competitor#1. You always provide argument from the facts or derivations based on the input information, and explicitly illustrate your NEW approach, fix and technique.\n <root><problem_statement>{problem}</problem_statement></root>"""
        #prompt2 = """You are Competitor #2. Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You NEVER agree with Competitor#1. You may take a step back to think through the problem again. You always provide the missing KEY technique or misunderstood points by Competitor#1, and explicitly illustrate your NEW approach, fix and technique. \n <root><problem_statement>{problem}</problem_statement></root>"""
        
        prompt1 = prompt1.format(problem=self.problem.problem_description)
        prompt2 = prompt2.format(problem=self.problem.problem_description)

        prompts = [prompt1, prompt2]
        messages1 = [{
                'role': 'user',
                'content': prompt1
            },
            {
                'role': 'assistant',
                'content': "understood."
            },
            ]
        messages2 = [{
                'role': 'user',
                'content': prompt2
            },
            {
                'role': 'assistant',
                'content': "understood."
            },
            ]
        messages = [messages1, messages2]

        step = 0
        id1, id2 = 1, 2
        while step < 5:
            step += 1
            self.logger.info(f"\n\n***************Step {step}: Competitor {id1} is running...***************\n\n")

            messages[id1-1].append(
                {
                    'role': 'user',
                    'content': prompts[id1-1]
                },
            )
            self.logger.info(f"Competitor#{id1} LLM Input: {messages[id1-1]}")
            
            out = self.llm.run_messages(messages=messages[id1-1], temperature=1)
            code = self.worker(out)
            self.logger.info(f"Step {step}: Competitor {id1}'s output is {out}")

            s = Solution(code, self.problem.problem_name, self.problem.sample_input_path, self.problem.sample_output_path, self.problem.full_input_path, self.model_name)
            testreport, fullreport = s.eval()
            self.sm.add_solution(s)
            if s.to_submit_signal:
               self.logger.info(f"Problem Solved!! ready for submit")
               break 
            
            if fullreport and testreport:
                self.logger.info(f"Step {step}: Competitor #{id1}'s testreport is {testreport.content} \n Full test report: {fullreport.message}\n")
            solution_list.append(s)
            
            messages[id1-1].append(
                {
                    'role': 'assistant',
                    'content': out
                },
            )
            
            prompts[id2-1] = f"##Competitor #{id1} provided this <competitor_{id1}_solution>{code}</competitor_{id1}_solution>\n ##The Evaluation Results of Competitor #{id1}'s solution:\n <sample_test>{testreport}</sample_test> <full_test>{fullreport}</full_test>"
            
            id1,id2 = id2,id1
        self.logger.info(f"{solution_list}\n")
            
        return solution_list
    
    def interpreter(self):
        """
        Prompt = "Rephrases the problem description for clearer understanding."
        """
        messages = [
            {
                'role': 'user',
                'content': prompt_rephrase_problem.format(problem=self.problem.problem_description)
            },
        ]
        out = self.llm.run_messages(messages=messages, temperature=0)
        self.logger.info(f"Rephraser output: {out}")
        self.problem.problem_description = out
        return out

    def meta_learning(self):
        """
        Preloads historical problem and solution messages for context.
        """
        problem_dir = "dataset/2023/practice/"
        problem_list = [
            "cheeseburger_corollary_ch1.md",
            "cheeseburger_corollary_ch2.md",
            "dim_sum_delivery.md",
            "two_apples_a_day.md",
            "road_to_nutella.md"
        ]
        for problem in problem_list:
            self.preload_messages.extend([
                {
                    'role': 'user',
                    'content': 'You are a world-class coding competitor solving complex problems. I will give you a problem statement, and you analyze the problem and provide a solution.',
                },
                {
                    'role': 'assistant',
                    'content': 'Understood.',
                },
                {
                    'role': 'user',
                    'content': Path(problem_dir + problem).read_text(),
                },
                {
                    'role': 'assistant',
                    'content': Path(problem_dir + problem[:-3] + '_sol.md').read_text(),
                },
            ])
        self.messages = self.preload_messages
        return self.preload_messages

    def reflection_pro(self,):
        """
        Reflects on the solution based on test reports and provides improvements.
        """
        solution_list = []

        code = self.solve_problem_pro()[0].code
        
        while self.reflection_step < 3:
            s = Solution(code, self.problem.problem_name, self.problem.sample_input_path, self.problem.sample_output_path, self.problem.full_input_path, self.model_name)
            testreport, full_testreport = s.eval()
            solution_list.append(s)
            
            self.messages.append(
                {
                    'role': 'user',
                    'content': REFLECTION_INSTRUCTIONS_USER.format(
                        incorrect_solution="[check above the solution and see the below report for reflection]",
                        test_report=testreport.content + "##FULL evaluation results:\n"+ full_testreport.content,
                    )
                })
            out = self.llm.run_messages(messages=self.messages)
            self.logger.info(f"Reflection output: {out}")
            self.messages.append(
                {
                    'role': 'assistant',
                    'content': out
                })
            
            self.reflection_step += 1
            code = self.worker(out)     
            if code:
                s = Solution(code, self.problem.problem_name, self.problem.sample_input_path, self.problem.sample_output_path, self.problem.full_input_path, self.model_name)
                solution_list.append(s)

        return solution_list
    
    def analyzer_provide_N(self,):
        return
    
    def solve_problem_pro(self,):
        """
        Solves the problem using a professional approach.
        """
        self.messages.append({
            'role': 'user',
            'content': initial_advisor_prompt.format(problem=self.problem.as_xml),
        })
        out = self.llm.run_messages(messages=self.messages)
        self.messages.append({
            'role': 'assistant',
            'content': out,
        })
        self.logger.info(f"Advisor output: {out}")
 
        code = self.worker(out)
        
        if code:
            s = Solution(code, self.problem.problem_name, self.problem.sample_input_path, self.problem.sample_output_path, self.problem.full_input_path, self.model_name)
        return [s]
    
    def chain_of_thoughts(self):
        """
        Uses a chain-of-thought process to solve the problem.
        """
        messages = []
        question_list = [
            "Carefully read through the problem statement line by line, and rephrase the problem for clearer understanding.\n\n##Problem:\n{problem}\n",
            "List all the key information of this problem for meta-learning: <label>[the keywords of the problem statement, including types of the problem, solution type, algorithms]</label>",
            "Analyze which solution is the most promising. Analyze the time complexity. **If this is a familiar problem, please don't be fooled by experience.** Note: Please be very careful about data contamination. Don't use your memory or knowledge to solve the problem. Read through the problem and provide a correct solution based on your reasoning only.",
            "Pick the best solution and implement the source code with comments on each line of the code."
        ]

        for question in question_list:
            formatted_question = question.format(problem=self.problem.as_xml)
            messages.append({
                'role': 'user',
                'content': formatted_question,
            })
            self.logger.info(formatted_question)
            out = self.llm.run_messages(messages=messages)
            messages.append({
                'role': 'assistant',
                'content': out,
            })
            self.logger.info(f"Assistant output: {out}")

        code = self.worker(out)
        if code:
        
            s = Solution(code, self.problem.problem_name, self.problem.sample_input_path, self.problem.sample_output_path, self.problem.full_input_path, self.model_name)
            testreport, full_testreport = s.eval()
            
        return [s]
    
    def worker(self, assistant_output):
        """
        Processes assistant output to extract and verify the source code.
        """
        messages = [
            {
                'role': 'user',
                'content': CODER_INSTRUCTIONS.format(code=assistant_output)
            },
        ]
        out = self.fast_llm.run_messages(messages=messages)
        
        code = extract_text(out, '<source_code>')
        code = maybe_remove_backticks(code)
    
        if verify_code_syntax(code):
            self.logger.info(f"Refactor the code=================\nCode syntax correct:\n{code}")
            return code
        else:
            return ""
            #raise ValueError("Source code is not compilable.")

def print_tree(node: Node | None, level: int = 0, prefix: str = ""):
    if node is None:
        return
    # Print current node with the appropriate prefix and score information
    connector = "└─" if level > 0 and not node.parent.children[-1] == node else "├─"
    print(f"{prefix}{connector} Node(state=node.state, Q={node.score}, visits={node.visits}, depth=)")
    # Update the prefix for children
    new_prefix = prefix + ("   " if connector == "└─" else "│  ")
    # Recursively print each child
    for idx, child in enumerate(node.children):
        is_last_child = idx == len(node.children) - 1
        if is_last_child:
            print_tree(child, level + 1, new_prefix)
        else:
            print_tree(child, level + 1, new_prefix)

if __name__ == '__main__':
    #problem_name = 'Prime Subtractorization'
    #problem_name = 'Subsonic Subway'
    #problem_name = 'Substantial Losses'

    #problem = load_problem_from_folder('2024', 'Round1/', problem_name, logger)
    #logger.info(f"Solving {problem_name}")
    logger = create_logger(f'logs/trainer.log', 'trainer')
    
    from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training
    
    problem_directory = "dataset/2023/round2"
    problem_names = list_problem_names(problem_directory, "2023")
    problem_list = []
    for problem_name in problem_names[:1]:
        problem = load_problem_training(problem_name, Path(problem_directory))
        problem_list.append(problem)
            
        model_name = 'gpt3.5' #ranking powerful to less ['o1', 'gpt4', 'claude', 'gemini', 'gpt3.5'] from most capable to least capable 
        #trainer1 = Trainer(model_name, problem,)
        #sols = trainer1.battle_ground()
        
        mcts = MCTS(model_name, problem)
        solution_node = mcts.mcts_trial(problem, max_steps=10)
        print(mcts.sm.solution_manager)
        #mcts.sm.to_submit('to_submit/')

        from mtcs_v2 import print_tree
        print_tree(mcts.root)
        


