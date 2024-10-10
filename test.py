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
    def __init__(self, model_name, problem):
        self.problem = problem
        self.model_name = model_name

        self.logger = create_logger(f'logs/{problem.problem_name}_{model_name}.log', '{problem.problem_name}_{model_name}')
        self.llm = LLM(model_name=model_name, logger=self.logger)
        self.messages = []
        self.reflection_step = 0
<<<<<<< Updated upstream
<<<<<<< Updated upstream
         
=======
        self.solution_list=[]

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

>>>>>>> Stashed changes
=======
    
    def run(self, method):
        if method == "reflection_pro":
            self.reflection_pro()
        elif method == "solve_problem_pro":
            self.solve_problem_pro()
        elif method == "chain_of_thoughts":
            self.chain_of_thoughts()
        else:
            raise ValueError("method not claimed")
        
>>>>>>> Stashed changes
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
                testreport, full_testreport = s.eval()
                solution_list.append(s)

        return solution_list
    
    def solve_problem_pro(self,):
        """
        Solves the problem using a professional approach.
        """
        self.messages.append({
            'role': 'user',
            'content': self.problem.as_xml,
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
            testreport, full_testreport = s.eval()
        
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
                'content': CODER_INSTRUCTIONS + f"This is the code: {assistant_output}"
            },
        ]
        out = self.llm.run_messages(messages=messages)
        
        code = extract_text(out, '<source_code>')
        code = maybe_remove_backticks(code)
    
        if verify_code_syntax(code):
            self.logger.info(f"Code syntax correct:\n{code}")
            return code
        else:
            return ""
            #raise ValueError("Source code is not compilable.")

if __name__ == '__main__':
    problem_name = 'Prime Subtractorization'

    logger = create_logger(f'logs/trainer.log', 'trainer')
    problem = load_problem_from_folder('2024', 'Round1/', problem_name, logger)
    logger.info(f"Solving {problem_name}")

    _ = output_format_indicator(problem, logger)
    
    sm = SolutionManager()
    
    model_name = 'gpt4'
    model_capability_ranking = 'gpt4' #['o1', 'gpt4', 'claude', 'gemini', gpt3.5] from most capable to least capable 
    trainer1 = Trainer(model_name, problem)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
    sols = trainer1.reflection_pro()
=======
    sols = trainer1.solve_problem_pro()
>>>>>>> Stashed changes
    for s in list(sols):
        sm.add_solution(s)

=======
    s = trainer1.solve_problem_pro()
    testreport, full_testreport = s.eval()
    sm.add_solution(s)
>>>>>>> Stashed changes
    sm.to_submit('to_submit/')


