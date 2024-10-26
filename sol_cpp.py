import asyncio
import multiprocessing
import threading
import subprocess
import datetime
import os
import math
from concurrent.futures import ThreadPoolExecutor
import sys
from logging import Logger
from pathlib import Path 
from utils import load_problem, create_logger, load_problem_from_folder
from utils import Problem, clean_code_string, TestReport
from lib.llms import LLM
from lib.prompts import * 
import time
import re
from typing import Any, List, Optional
from pydantic import BaseModel, Field
import aiofiles
import traceback


def verify_code_syntax(code_str, selected_language):
    if selected_language == "python":
        try:
            compile(code_str, '<string>', 'exec')
            return True
        except SyntaxError as e:
            return False

    elif selected_language == "cpp":
        return True
        # try: # we may be able to combine this with other parts to save time for cpp as we write the file again
        #     cpp_file = "temp_code.cpp"
        #     with open(cpp_file, 'w') as f:
        #         f.write(code_str)
            
        #     # Try to compile the C++ code using g++
        #     result = subprocess.run(["g++", "-fsyntax-only", cpp_file], 
        #                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        #     # Remove the temporary file
        #     os.remove(cpp_file)
            
        #     # Return True if compilation was successful, False otherwise
        #     return result.returncode == 0 # True

        # except Exception as e:
        #     return False
    
    else:
        raise ValueError("Unsupported language. Choose either 'python' or 'cpp'.")
    
def extract_text(input_string, format):
    # Use a regex pattern to extract text between <prompt> and </prompt>
    #match = re.search(f'{format}(.*?){format.replace('<','</')}', input_string, re.DOTALL)
    match = re.search(f'{format}(.*?){format.replace("<", "</")}', input_string, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return None

def maybe_remove_backticks(solution: str) -> str:
    "Remove backticks from the solution"
    solution = solution.strip()
    solution = re.sub(r'^```python\s*', '', solution) # TODO to add CPP (check the format from log)
    solution = re.sub(r'\s*```$', '', solution)

    # Remove backticks for C++ code block
    solution = re.sub(r'^```cpp\s*', '', solution)
    solution = re.sub(r'^```c\+\+\s*', '', solution)
    return solution

def save_to_disk(content: str, path: Path, logger: Logger):
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"Starts to write to {path}.")
    with path.open("w", encoding ="utf-8") as f:
        f.write(content)
    if logger:
        logger.info(f"Finish writing to {path}.")

# Run coroutine in a thread-safe manner
def run_coroutine(coro):
    try:
        # Check if the thread has an event loop, if not, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

class Trainer():
    def __init__(self, problem, logger, zero_shot_llm_model_name, extract_code_llm_model_name, reflection_llm_model_name=None, advisor_llm_model_name=None):
        self.zero_shot_llm = LLM(model_name=zero_shot_llm_model_name, logger=logger)
        self.zero_shot_llm_model_name = zero_shot_llm_model_name

        self.extract_code_llm = LLM(model_name=extract_code_llm_model_name, logger=logger)
        self.extract_code_llm_model_name = extract_code_llm_model_name
        
        # if use reflection and advisor
        if reflection_llm_model_name is not None:
            self.reflection_llm = LLM(model_name=reflection_llm_model_name, logger=logger)
            self.reflection_llm_model_name = reflection_llm_model_name
        else:
            self.reflection_llm = None
            self.reflection_llm_model_name = None
        
        if advisor_llm_model_name is not None:
            self.advisor_llm = LLM(model_name=advisor_llm_model_name, logger=logger)
            self.advisor_llm_model_name = advisor_llm_model_name
        else:
            self.advisor_llm = None
            self.advisor_llm_model_name = None
        
        self.messages = []
        self.current_solutions = ""
        self.reflection_step = 0
        self.problem = problem
        self.preload_messages = []
        self.logger = logger

    def interpreter(self,) -> dict:
        # zero-shot code
        messages = [
            {
                'role': 'user',
                'content': prompt_rephrase_problem.format(problem=self.problem.problem_description)
            },
            ]
        out = self.zero_shot_llm.run_messages(messages=messages, temperature=0)
        self.logger.info(f"rephraser output: {out}")

        self.problem_description = out

        return out
        

    def historian_way(self):
        problem_dir = "dataset/2023/practice/"
        problem_list = ["cheeseburger_corollary_ch1.md", "cheeseburger_corollary_ch2.md", "dim_sum_delivery.md", "two_apples_a_day.md", "road_to_nutella.md"]
        for problem in problem_list:
            self.preload_messages += [
                {
                    'role': 'user',
                    'content': 'You are a world-class coding competitor solving complex problems. I will give you problem statement and you analyze the problem and provide solution.',
                },
                {
                    'role': 'assistant',
                    'content': 'understood.',
                },
                
                {
                    'role': 'user',
                    'content': Path(problem_dir+problem).read_text(),
                },
                {
                    'role': 'assistant',
                    'content': Path(problem_dir+problem[:-3]+'_sol.md').read_text(),
                },
            ]
        self.messages = self.preload_messages
        return self.preload_messages


    def battle_ground(self, selected_language, total_iter=4): # TODO
        
        solution_list = []

        #'Subsonic Subway'
        prompt1 = """You are Competitor #1. Your goal is to provide the TRULY correct and NO-TIMEOUT {selected_language} solution. You NEVER agree with Competitor#2. You always provide the missing or misunderstood points by Competitor#2 and provide argument from the facts or derivations based on the input information. \n <root><problem_statement>{problem}</problem_statement></root>"""
        prompt2 = """You are Competitor #2: Your goal is to provide the TRULY correct and NO-TIMEOUT {selected_language} solution. You MAY learn from Competitor#1 but you always provide argument from the facts or derivations based on the input information. \n <root><problem_statement>{problem}</problem_statement></root>"""
        
        # #'Prime Subtractorization'
        # prompt1 = """You are Competitor #1: Your goal is to provide the TRULY correct and NO-TIMEOUT {selected_language} solution. You NEVER agree with Competitor#2. You always provide argument from the facts or derivations based on the input information, and explicitly illustrate your NEW approach, fix and technique.\n <root><problem_statement>{problem}</problem_statement></root>"""
        # prompt2 = """You are Competitor #2. Your goal is to provide the TRULY correct and NO-TIMEOUT {selected_language} solution. You NEVER agree with Competitor#1. You always provide the overlook insights from the problem, provide NEW approach, provide fix and advanced techniques. \n <root><problem_statement>{problem}</problem_statement></root>"""
        
        #prompt1 = """You are Competitor #1: Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You NEVER agree with Competitor#2. You always provide argument from the facts or derivations based on the input information, and explicitly illustrate your NEW approach, fix and technique.\n <root><problem_statement>{problem}</problem_statement></root>"""
        #prompt2 = """You are Competitor #2. Your goal is to provide the TRULY correct and NO-TIMEOUT solution. You NEVER agree with Competitor#1. You may take a step back to think through the problem again. You always provide the missing KEY technique or misunderstood points by Competitor#1, and explicitly illustrate your NEW approach, fix and technique. \n <root><problem_statement>{problem}</problem_statement></root>"""
        
        prompt1 = prompt1.format(
            problem=self.problem.problem_description, 
            selected_language=selected_language)
        prompt2 = prompt2.format(
            problem=self.problem.problem_description,
            selected_language=selected_language)

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
        while step < total_iter: # TODO setting for how many interations in total two competitors can have
            step += 1
            self.logger.info(f"\n\n***************Step {step}: Competitor {id1} is running...***************\n\n")

            messages[id1-1].append(
                {
                    'role': 'user',
                    'content': prompts[id1-1]
                },
            )
            self.logger.info(f"Competitor#{id1} LLM Input: {prompts[id1-1]}")
            
            out = self.zero_shot_llm.run_messages(messages=messages[id1-1], temperature=1) # using zero_shot_llm initatied model
            code = self.worker(out, selected_language)
            self.logger.info(f"Step {step}: Competitor {id1}'s output is {out}")

            # evaluate against sample output
            test_report_sample = self.evaluator(code, selected_language)

            if test_report_sample.success_rate_number != 1:
            
                messages[id1-1].append(
                    {
                        'role': 'assistant',
                        'content': out
                    },
                )
                
                prompts[id2-1] = f"##Competitor #{id1} provided this <competitor_{id1}_solution>{code}</competitor_{id1}_solution>\n ##The Evaluation Results of Competitor #{id1}'s solution:\n <sample_test>{test_report_sample}</sample_test>"
                
                id1,id2 = id2,id1

        return code # code, response


    def reflection_agent(self, solution, test_report_str, selected_language):
        reflection_message = REFLECTION_INSTRUCTIONS_USER.format(
                    incorrect_solution = solution,
                    test_report = test_report_str,
                    selected_language = selected_language)
        
        self.messages += [
            {
                'role': 'user',
                'content': reflection_message
            },
        ]
        self.logger.info(f"Reflection step: {self.reflection_step} ")
        out = self.reflection_llm.run_messages(messages=self.messages, temperature=0) # run_messages has self.logger for details
        self.reflection_step += 1
        
        return self.worker(out, selected_language), out # call llm to extract code in worker

    def timeout_agent(self, code, selected_language):
        timeout_agent_message = timeout_agent_prompt.format(
                    problem=self.problem.as_xml,
                    timeout_solution = code,
                    selected_language = selected_language)
        
        messages = [ # a clean one
            {
                'role': 'user',
                'content': timeout_agent_message
            },
        ]
        out = self.reflection_llm.run_messages(messages=messages, temperature=0.7) # run_messages has self.logger for details
        
        return self.worker(out, selected_language) #, out # call llm to extract code in worker


    # if score == 1, check for constrains and corner cases
    def last_check_agent(self, solution, selected_language):
        last_check_message = LAST_CHECK_INSTRUCTIONS_USER.format(
                    correct_solution = solution,
                    selected_language = selected_language)
        self.messages += [
            {
                'role': 'user',
                'content': last_check_message
            },
        ]
        self.logger.info(f"Last check. Reflection step: {self.reflection_step}")
        out = self.reflection_llm.run_messages(messages=self.messages, temperature=0)
        self.reflection_step += 1
        
        return self.worker(out, selected_language), out # call llm to extract code in worker
    

    def solve_problem_pro(self, selected_language) -> dict:
        
        self.messages += [
            {
                'role': 'user',
                'content': simple_initial_advisor_prompt.format(
                    problem=self.problem.as_xml,
                    selected_language=selected_language) # self.problem.as_xml
            } 
        ]
        out = self.zero_shot_llm.run_messages(messages=self.messages) # includes logger
        
        self.messages += [{
                'role': 'assistant',
                'content': out,
            },]
        return self.worker(out, selected_language), out
    

    def o1_solve_problem_pro(self, selected_language) -> dict:
        
        self.messages += [
            {
                'role': 'user',
                'content': o1_simple_initial_advisor_prompt.format(
                    problem=self.problem.as_xml,
                    selected_language=selected_language)
            } 
        ]
        out = self.zero_shot_llm.run_messages(messages=self.messages) # includes logger
        
        self.messages += [{
                'role': 'assistant',
                'content': out,
            },]
        return self.worker(out), out


    def solve_problem_pre_rag_code(self, selected_language):
        # pre-rag 1/2: generate code
        self.messages += [ # self.messages
            {
                'role': 'user',
                'content': solve_problem_pre_rag_prompt_code.format(
                    problem=self.problem.as_xml,
                    selected_language=selected_language)
            } 
        ]

        out_code = self.zero_shot_llm.run_messages(messages=self.messages) # includes logger
        self.messages += [{
                'role': 'assistant',
                'content': out_code,
            },]

        return self.worker(out_code, selected_language)


    def solve_problem_pre_rag_algorithm(self, selected_language):
        # pre-rag 2/2: generate algorithm for rag search
        messages = [
            {
                'role': 'user',
                'content': solve_problem_pre_rag_prompt_algorithm.format(
                    problem=self.problem.as_xml)
            } 
        ]
        out_algorithm = self.extract_code_llm.run_messages(messages=messages) # includes logger

        # below extract code and algorithm using smaller models and return
        return self.worker_algorithm(out_algorithm)

    
    def solve_problem_rag(self, selected_language, algorithmic_strategies, alike_solutions_and_codes):
        # messages = []
        self.messages += [ # self.messages
            {
                'role': 'user',
                'content': solve_problem_rag_prompt.format(
                    # problem=self.problem.as_xml, # as included in self.messages already
                    selected_language=selected_language,
                    algorithmic_strategies=algorithmic_strategies,
                    alike_solutions_and_codes=alike_solutions_and_codes)
            } 
        ]
        out = self.reflection_llm.run_messages(messages=self.messages) # includes logger
        
        self.messages += [{
                'role': 'assistant',
                'content': out,
            },]
        
        return self.worker(out, selected_language)
    
    def chain_of_thoughts(self):
        messages = []
        question0 = """Carefully read through the problem statement line by line, and rephrase the problem for clearer understanding.\n\n ##Problem:{problem}\n"""
        question0 = question0.format(problem=self.problem.as_xml)
        question1 = "list all the key information of this problem for meta-learning: <label>[the keywords of the problem statement, including types of the problem, solution type, algorithms]</label>"
        question2 = "Analyze which solution is the most promising correct solution. Analyze the time complexity. **If this is a familiar problem, please don't be fooled by the experience. \n**Note please be very careful about data contamination. Don't use your memory or knowledge to solve the problem. Read through the problem and provide correct solution based on the your reasoning only."
        question3 = "pickup the best solution and implement the source code with the comments on each line of the code."

        question_list = [question0, ] #question1, question2, question3,

        for idx, question in enumerate(question_list):
            messages += [
                {
                    'role': 'user',#self.problem.as_xml, #
                    'content': question,#initial_advisor_prompt.format(problem=self.problem.as_xml,)
                }
            ]
            self.logger.info(question)
            if idx == 0:
                out = self.zero_shot_llm.openai_ft_messages(messages=messages)
            else:
                out = self.zero_shot_llm.openai_messages(messages=messages)
            self.messages += [{
                    'role': 'assistant',
                    'content': out,
                },]
            # self.logger.info(out)
            # self.logger.info(f"advisor output: {out}")
        return self.worker(out), out
      
    
    def manager(self, ):
        # zero-shot code
        messages = [
            {
                'role': 'user',
                'content': initial_advisor_prompt
            },
            {
                'role': 'user',
                'content': problem.as_xml
            },
            {
                'role': 'user',
                'content': manager_prompt
            },
            {
                'role': 'assistant',
                'content': 'understood. Please provide the current status of all solutions you have tried.'
            },
            {
                'role': 'user',
                'content': self.current_solutions
            },
        ]
        new_instruction = self.zero_shot_llm.run_messages(messages=messages)
        self.logger.info(f"manager output: {new_instruction}")
        
        return new_instruction
        
    # worker to extract code
    def worker(self, out0, selected_language):
        extract_code_message = EXTRACT_CODE_PROMPT.format(
            output=out0,
            selected_language=selected_language)
        
        messages1 = [
            {
                'role': 'user',
                'content': extract_code_message
            }
        ]

        code = self.extract_code_llm.run_messages(messages=messages1)
        
        # self.logger.info(f"EXTRACT_CODE_PROMPT output:\n{code}")
        
        # may not need it - if still <source_code> in out
        if '<source_code>' in code:
            code = extract_text(code, '<source_code>')

        code = maybe_remove_backticks(code)

        if code:
            if verify_code_syntax(code, selected_language):
                self.logger.info(f"code syntax correct:\n{code}")
                return code
            else:
                raise ValueError("source code is not compilable")
        else:
            raise ValueError("no source code generated")

    # worker to extract list of algorithm
    def worker_algorithm(self, out0):

        extract_algorithm_message = EXTRACT_ALGORITHM_PROMPT.format(
            output=out0)

        messages2 = [
            {
                'role': 'user',
                'content': extract_algorithm_message
            }
        ]
        out = self.extract_code_llm.run_messages(messages=messages2)
        return out


    # evaluate + save the interim solutions in generated_folder
    def evaluator(self, code, selected_language):
        try:
            with ThreadPoolExecutor(max_workers=1) as executor: #  TODO timeout
                if selected_language == "python":
                    future_report = executor.submit(run_coroutine, check_correctness(self.problem, code, self.problem.sample_input, self.problem.sample_output, 3, selected_language))
                    test_report_sample = future_report.result()
                    # self.current_solutions += '\n'+ test_report_sample.content + '\n\n'

                elif selected_language == "cpp":
                    sample_output_file = self.generate_sample_cpp(code, selected_language)
                    test_report_sample = check_output(self.problem.sample_input_path, sample_output_file, self.problem.sample_output_path)
                    #check_output(input_data_file, model_output_file, expected_output_file)
                
                self.logger.info(f"test report samples: {test_report_sample.content}")

                # update score and code
                current_score = test_report_sample.success_rate_number
                self.problem.latest_score = current_score

                if self.problem.best_code == "": # if first-time (best_score == ""), update the recent score
                    self.problem.best_code = code
                    self.problem.best_score = current_score
                
                elif current_score > self.problem.best_score: # only update when current score better than saved best_score
                    self.problem.best_score = current_score
                    self.problem.best_code = code

                # save interim codes files in generated_folder; update latest_code_path
                # TODO consider the time consumption but this helps to select all solutions instead of the latest ones
                current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
                if selected_language == "python":
                    py_file = self.problem.generated_folder / f"{self.problem.problem_name.replace(' ', '_')}_{current_score:.1f}_{current_time}.py"
                    self.problem.latest_code_path = py_file
                    save_to_disk(code, py_file, self.logger)
                elif selected_language == "cpp":
                    cpp_file = self.problem.generated_folder / f"{self.problem.problem_name.replace(' ', '_')}_{current_score:.1f}_{current_time}.cpp"
                    self.problem.latest_code_path = cpp_file
                    save_to_disk(code, cpp_file, self.logger)
                
                return test_report_sample
        
        except Exception as e:
            self.logger.error(f"Error in evaluator: {e}")
            raise
        
    def generate_sample_cpp(self, code, selected_language):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_report = executor.submit(run_coroutine, write_sample_output(1, code, self.problem, selected_language, timeout=30, logger=self.logger))
            sample_output_file = future_report.result()
            self.logger.info(f"write the full out to: {sample_output_file}")
            return sample_output_file


    def generate_full(self, code, selected_language):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_report = executor.submit(run_coroutine, write_full_output(1, code, self.problem, selected_language, self.logger))
            # try:
            full_output_file = future_report.result()
            self.logger.info(f"write the full out to: {full_output_file}")
            return full_output_file
            # Oyiyi 10/19            
            # except subprocess.TimeoutExpired as e:
            #     # self.logger.error(f"TimeoutExpired while generating full output: {str(e)}")
            #     raise e  # Propagate the exception to the main function
            # except RuntimeError as e:
            #     self.logger.error(f"RuntimeError while generating full output: {str(e)}")
            #     raise e  # Propagate the exception to the main function
            # except Exception as e:
            #     self.logger.error(f"An unexpected error occurred in generate_full: {str(e)}")
            #     raise e  # Propagate other unexpected exceptions
        

    def o1_generate_full(self, code):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_report = executor.submit(run_coroutine, o1_write_full_output(1, code, self.problem, self.logger))
            full_output_file = future_report.result()
            self.logger.info(f"write the full out to: {full_output_file}")
            return full_output_file


# python: run the program code and compare the sample outputs
async def exec_program(problem, program, input_data, expected_output, timeout):
    total = int(input_data.split("\n")[0])
    starting_timer = time.time()
    if not program:
        return TestReport(
            total=total,
            failed= 0,
            success_rate = format(0, ".0%"),
            success_rate_number = 0.0,
            success_rate_full = format(0, ".0%"),
            total_full=total,
            failed_full=0,    
            status="error", 
            message=f"the source code is empty",
            output=f""
        )
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", program,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input=input_data.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return TestReport(
                total=total,
                failed= 0,
                success_rate = format(0, ".0%"),
                success_rate_number = 0.0,
                success_rate_full = format(0, ".0%"),
                total_full=total,
                failed_full=0,             
                status="timeout",
                message=f"Took too long! Your program timed out after {timeout} seconds of execution.",
                output=f""
            )
        
        if process.returncode != 0:
            return TestReport(
                total=total,
                failed= 0,
                success_rate = format(0, ".0%"),
                success_rate_number = 0.0,
                success_rate_full = format(0, ".0%"),
                total_full=total,
                failed_full=0,    
                status="error", 
                message=f"Program execution failed: {stderr.decode()}",
                output=f""
            )
        else:
            if stdout.decode().strip() == expected_output.strip():
                return TestReport(
                    total=total,
                    failed= 0,
                    success_rate = format(1, ".0%"),
                    success_rate_number = 1.0,
                    success_rate_full = format(1, ".0%"),
                    total_full=total,
                    failed_full=0,    
                    status='passed', 
                    message=f"The program successfully pass {total} sample results with a time consumption of {time.time()-starting_timer}ms",
                    output=f"{stdout.decode()}",
                )
            else:
                actual_output = stdout.decode()
                actual_output_cases = actual_output.split("\n")
                expected_output_cases = expected_output.split("\n")
                success = failed = 0
                # failure_case = [];
                # expected_case = [];
                if len(actual_output_cases) == 0:
                    return TestReport(
                        total=total,
                        failed= 0,
                        success_rate = format(0, ".0%"),
                        success_rate_full = format(0, ".0%"),
                        success_rate_number = 0.0,
                        total_full=total,
                        failed_full=0,              
                        status="error",
                        message=f"There is no output generated from the source code.",
                        output=f"",
                    )
                for i in range(len(actual_output_cases)):
                    # if not empty "" which can't do split(":")
                    if actual_output_cases[i].strip() and expected_output_cases[i].strip():
                    # if both numbers, compare up to 1e-7 decimals
                        try:
                            expected_value = float(expected_output_cases[i].split(":")[1].strip()) # "Case #1: 20.710678118654748" -> 20.71067811865474
                            actual_value = float(actual_output_cases[i].split(":")[1].strip())
                            
                            if math.isclose(expected_value, actual_value, abs_tol=1e-7):
                                success += 1
                            else:
                                failed += 1

                        # if not numbers, perform a direct string comparison
                        except ValueError:
                            
                            if (expected_output_cases[i] == actual_output_cases[i]):
                                success = success + 1
                            else:
                                failed = failed + 1

                # success_rate = format(success/len(expected_output_cases), ".0%")
                # success_rate_number = float(success/len(expected_output_cases))
                success_rate = format(success/total, ".0%")
                success_rate_number = float(success/total)

                if len(expected_output_cases) > 10:
                    message = "generated output is not correct."
                else:
                    message = f"<expected>\n{expected_output}</expected>\n---\n<got>\n{stdout.decode()}</got>"
                return TestReport(
                    total=total,
                    failed= failed,
                    success_rate = success_rate,
                    success_rate_full = success_rate,
                    success_rate_number = success_rate_number,
                    total_full=total,
                    failed_full=failed,    
                    status='FAILED',
                    message=message,
                    output=f"{stdout.decode()}",
                )
    except Exception:
        return TestReport(
            total=total,
            failed= 0,
            success_rate = format(0, ".0%"),
            success_rate_full = format(0, ".0%"),
            success_rate_number = 0.0,
            total_full=total,
            failed_full=0,              
            status="error",
            message=f"An error occurred: {traceback.format_exc()}",
            output=f"",
        )


# cpp: run the program code and compare the sample outputs TODO not being used
async def exec_program_cpp(problem, program: str, input_data: str, expected_output: str, timeout: float) -> TestReport:
    total = int(input_data.split("\n")[0])
    starting_timer = time.time()

    if not program:
        return TestReport(
            total=total,
            failed=0,
            success_rate=format(0, ".0%"),
            success_rate_number=0.0,
            success_rate_full=format(0, ".0%"),
            total_full=total,
            failed_full=0,
            status="error",
            message="The source code is empty.",
            output="",
        )

    # Define temporary file names
    cpp_file = Path(f"{problem.problem_name}_temp.cpp")
    exe_file = cpp_file.with_suffix('.exe')  # For Windows, use '.exe'; for Unix, use ''

    try:
        # Step 1: Write the C++ code to a temporary file
        async with aiofiles.open(cpp_file, 'w') as f:
            await f.write(program)

        # Step 2: Compile the C++ code using subprocess.run
        compile_command = f'g++ -std=c++17 -O2 "{cpp_file}" -o "{exe_file}"'

        try:
            compile_result = subprocess.run(
                compile_command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                env=os.environ.copy()
            )
        except subprocess.TimeoutExpired:
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_number=0.0,
                success_rate_full=format(0, ".0%"),
                total_full=total,
                failed_full=0,
                status="timeout",
                message=f"Compilation timed out after {timeout} seconds.",
                output="",
            )
        except subprocess.CalledProcessError as e:
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_number=0.0,
                success_rate_full=format(0, ".0%"),
                total_full=total,
                failed_full=0,
                status="error",
                message=f"Compilation failed: {e.stderr.decode().strip()}",
                output="",
            )

        # Check if the executable file was created
        if not exe_file.exists():
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_number=0.0,
                success_rate_full=format(0, ".0%"),
                total_full=total,
                failed_full=0,
                status="error",
                message=f"Compilation succeeded but executable not found: {exe_file}",
                output="",
            )

        # Step 3: Execute the compiled binary using subprocess.run
        try:
            execution_result = subprocess.run(
                [str(exe_file)],
                input=input_data.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=True,
                env=os.environ.copy()
            )
        except subprocess.TimeoutExpired:
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_number=0.0,
                success_rate_full=format(0, ".0%"),
                total_full=total,
                failed_full=0,
                status="timeout",
                message=f"Execution timed out after {timeout} seconds.",
                output="",
            )
        except subprocess.CalledProcessError as e:
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_number=0.0,
                success_rate_full=format(0, ".0%"),
                total_full=total,
                failed_full=0,
                status="error",
                message=f"Program execution failed: {e.stderr.decode().strip()}",
                output="",
            )

        # Step 4: Compare the output
        actual_output = execution_result.stdout.decode().strip()
        expected_output = expected_output.strip()

        if actual_output == expected_output:
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(100, ".0%"),
                success_rate_number=1.0,
                success_rate_full=format(100, ".0%"),
                total_full=total,
                failed_full=0,
                status='passed',
                message=f"The program successfully passed {total} sample results with a time consumption of {int((time.time() - starting_timer) * 1000)}ms.",
                output=actual_output,
            )
        else:
            actual_output_cases = actual_output.split("\n")
            expected_output_cases = expected_output.split("\n")
            success = failed = 0

            if len(actual_output_cases) == 0:
                return TestReport(
                    total=total,
                    failed=0,
                    success_rate=format(0, ".0%"),
                    success_rate_full=format(0, ".0%"),
                    success_rate_number=0.0,
                    total_full=total,
                    failed_full=0,
                    status="error",
                    message="There is no output generated from the source code.",
                    output="",
                )

            for actual, expected in zip(actual_output_cases, expected_output_cases):
                actual = actual.strip()
                expected = expected.strip()
                if not actual or not expected:
                    continue
                try:
                    expected_value = float(expected.split(":")[1].strip())
                    actual_value = float(actual.split(":")[1].strip())
                    if math.isclose(expected_value, actual_value, abs_tol=1e-7):
                        success += 1
                    else:
                        failed += 1
                except (IndexError, ValueError):
                    if actual == expected:
                        success += 1
                    else:
                        failed += 1

            success_rate = format(success / total * 100, ".0%")
            success_rate_number = success / total

            if len(expected_output_cases) > 10:
                message = "Generated output is not correct."
            else:
                message = f"<expected>\n{expected_output}</expected>\n---\n<got>\n{actual_output}</got>"

            return TestReport(
                total=total,
                failed=failed,
                success_rate=success_rate,
                success_rate_full=success_rate,
                success_rate_number=success_rate_number,
                total_full=total,
                failed_full=failed,
                status='FAILED',
                message=message,
                output=actual_output,
            )

    except Exception:
        return TestReport(
            total=total,
            failed=0,
            success_rate=format(0, ".0%"),
            success_rate_full=format(0, ".0%"),
            success_rate_number=0.0,
            total_full=total,
            failed_full=0,
            status="error",
            message=f"An error occurred: {traceback.format_exc()}",
            output="",
        )

    finally:
        # Clean up temporary files
        for file in [cpp_file, exe_file]:
            try:
                if file.exists():
                    file.unlink()
            except Exception as e:
                print(f"Failed to delete temporary file {file}: {e}")


# ad-hoc purpose to check generated_solution txt vs txt/out
def check_output(input_data_file, model_output_file, expected_output_file):
    starting_timer = time.time()

    expected_output = Path(expected_output_file).read_text()
    actual_output = Path(model_output_file).read_text()
    actual_output_cases = actual_output.split("\n")
    expected_output_cases = expected_output.split("\n")
    success = failed = 0

    input_data = Path(input_data_file).read_text()
    total = int(input_data.split("\n")[0])

    if len(actual_output_cases) == 0:
        return TestReport(
            total=total,
            failed= 0,
            success_rate = format(0, ".0%"),
            success_rate_full = format(0, ".0%"),
            success_rate_number = 0.0,
            total_full=total,
            failed_full=0,              
            status="error",
            message=f"There is no output generated from the source code.",
            output=f"",
        )

    for i in range(len(actual_output_cases)):
        # if not empty "" which can't do split(":")
        if actual_output_cases[i].strip() and expected_output_cases[i].strip():
        # if both numbers, compare up to 1e-7 decimals
            try:
                expected_value = float(expected_output_cases[i].split(":")[1].strip()) # "Case #1: 20.710678118654748" -> 20.71067811865474
                actual_value = float(actual_output_cases[i].split(":")[1].strip())

                if math.isclose(expected_value, actual_value, abs_tol=1e-7):
                    success += 1
                else:
                    failed += 1

            # if not numbers, perform a direct string comparison
            except ValueError:
                
                if (expected_output_cases[i] == actual_output_cases[i]):
                    success = success + 1
                else:
                    failed = failed + 1


    success_rate = format(success/total, ".0%")
    success_rate_number = float(success/total)

    return TestReport(
        total=total,
        failed= failed,
        success_rate = success_rate,
        success_rate_full = success_rate,
        success_rate_number = success_rate_number,
        total_full=total,
        failed_full=failed,    
        status='FAILED',
        message="", #message,
        output="" #f"{stdout.decode()}",
    )


# includes generate output using code, compare it with the expected output
async def check_correctness(problem, program: str, input_data: str, expected_output: str, timeout: float, selected_language: str) -> TestReport:
    if selected_language == "python":
        return await exec_program(problem, program, input_data, expected_output, timeout)
    # elif selected_language == "cpp":
        # return await exec_program_cpp(problem, program, input_data, expected_output, timeout)


# generate full in python
async def generate_output_async(current_time, problem, program: str, input_file: Path, output_file: Path, timeout: int, logger: Logger):
    save_to_disk(program, problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.latest_score:.1f}_python_{current_time}.py", logger)
    # Open the input file for reading and output file for writing
    async with aiofiles.open(input_file, 'r') as infile, aiofiles.open(output_file, 'w') as outfile:
        # Read the input file content
        input_data = await infile.read()

        # Start the subprocess, providing input_data manually and capturing stderr
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", program,
            stdin=asyncio.subprocess.PIPE,  # Pass input via stdin
            stdout=asyncio.subprocess.PIPE,  # Capture stdout via PIPE
            stderr=asyncio.subprocess.PIPE
        )

        try:
            # Send input to the process and wait for the result
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_data.encode()),  # Pass input data as bytes
                timeout=timeout
            )

        except asyncio.TimeoutError:
            process.kill()
            logger.warning(f"Process timed out after {timeout} seconds")
            return str(output_file)

        # Check the return code after the process completes
        return_code = process.returncode
        if return_code != 0:
            logger.warning(f"Process exited with non-zero return code {return_code}")
            logger.error(f"Error: {stderr.decode().strip()}")
            return str(output_file)

        # Write the stdout output to the output file
        await outfile.write(stdout.decode())

        logger.info(f"Process completed successfully. Output written to {output_file}")

        return str(output_file)  # Return the path to the output file


# generate full in cpp
async def generate_output_cpp_async(current_time, problem, program: str, input_file: Path, output_file: Path, timeout: int, logger: Logger):
    # current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    exe_file = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.latest_score:.1f}_cpp_{current_time}.exe"
    cpp_file = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.latest_score:.1f}_cpp_{current_time}.cpp"
    save_to_disk(program, cpp_file, logger) # save the .cpp file as we save the output. this is the last code

    # should create and save the file .exe
    # compile_command = f'g++ -std=c++17 -O2 "{cpp_file}" -o "{exe_file}"' 
    compile_command = f'g++ -std=c++17 -O2 "{cpp_file}" -o "{exe_file}" -Wl,-stack_size,0x40000000' # 20m for 512MB/40m for 1G for stack overflow

    # into below to capture the error messages
    try:
        result = subprocess.run(
            compile_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Compilation succeeded: {result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Compilation failed: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to compile C++ code: {e.stderr.decode()}")

    # muted Oyiyi 10/18 night
    # try:
    #     _ = subprocess.run(
    #         compile_command, shell=True, check=True, stderr=subprocess.DEVNULL
    #     )
    # except subprocess.CalledProcessError as e:
    #     logger.error(f"Compilation failed: {e.stderr.decode()}")
    #     raise RuntimeError(f"Failed to compile C++ code: {e.stderr.decode()}")

    problem_name = cpp_file.stem.rsplit("_", 1)[0]

    try:
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            _ = subprocess.run(
                exe_file,
                stdin=infile,
                stdout=outfile,
                check=True,
                timeout=timeout,
            )
    # Oyiyi muted these two 10/19 let it naturally propagate
    # except subprocess.TimeoutExpired as e:
    #     logger.error(f"Execution timed out after {timeout} seconds for {problem_name}")
    #     # Handle timeout: decide if you want to retry, log a specific message, or return a default output
    #     raise e  # or some other fallback value

    # except subprocess.CalledProcessError as e:
    #     logger.error(f"Execution failed for {problem_name}: {e.stderr.decode()}")
    #     # Handle the failure of the executable: you can choose to retry or log and move on
    #     raise RuntimeError(f"Execution failed for {problem_name}: {e.stderr.decode()}") from e

    except Exception as e:
        logger.error(f"An unexpected error occurred during the execution of {problem_name}: {str(e)}")
        raise e

    return str(output_file) # Return the path to the output file. Generated results


# saved both .py/.cpp and .txt file to generated folder
async def write_sample_output(stage, source_code, problem, selected_language, timeout = 30, logger = None): 
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    sample_output_file = problem.generated_folder / f"{problem.problem_name.replace(' ', '_')}_{selected_language}_{current_time}.txt"
    
    if selected_language == "python": # below function saved .py/.cpp and output .txt
        output = await generate_output_async(current_time, problem, source_code, problem.sample_input_path, sample_output_file, timeout, logger)  # timeout in secs
    elif selected_language == "cpp":
        output = await generate_output_cpp_async(current_time, problem, source_code, problem.sample_input_path, sample_output_file, timeout, logger)  # timeout in secs

    if output:
        logger.info(f'successful generated the most recent sample output: {output}')

    return sample_output_file


# saved both .py/.cpp and .txt file to to_submit folder
async def write_full_output(stage, source_code, problem, selected_language, logger = None): 
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    full_output_file = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.latest_score:.1f}_{selected_language}_{current_time}.txt"
    
    # try:
    if selected_language == "python": # below function saved .py/.cpp and output .txt
        output = await generate_output_async(current_time, problem, source_code, problem.full_input_path, full_output_file, 30, logger)  # timeout in secs
    elif selected_language == "cpp":
        output = await generate_output_cpp_async(current_time, problem, source_code, problem.full_input_path, full_output_file, 30, logger)  # timeout in secs
    # Oyiyi 10/19
    # except subprocess.TimeoutExpired as e:
    #     # logger.warning("Timeout occurred during generate_full")
    #     raise e
    
    if output:
        logger.info(f'successful generated the most recent full output: {output}')

    # if latest one is not the best score, also generate best scored code and full output
    if problem.latest_score != problem.best_score:
        source_code = problem.best_code
        save_to_disk(source_code, problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.best_score}_{current_time}.py", logger) # TODO is this double used?
        full_output_file_best_score = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.best_score}_{current_time}.txt"
        
        output_best_score = await generate_output_async(current_time, source_code, problem.full_input_path, full_output_file_best_score, timeout=30, logger=logger)  # timeout in secs
        if output_best_score:
            logger.info(f'successful generated the best scored full output: {output_best_score}')

    return full_output_file


async def o1_write_full_output(stage, source_code, problem, logger = None): # TODO here saving the files
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    save_to_disk(source_code, problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_o1_{problem.latest_score}_{current_time}.py", logger) # diff here - added model to mark o1
    full_output_file = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_o1_{problem.latest_score}_{current_time}.txt"
    
    output = await generate_output_async(source_code, problem.full_input_path, full_output_file, 45, logger)  # timeout in secs
    if output:
        logger.info(f'successful generated the most recent full output: {output}')

    # TODO if latest one is not the best score, also generate best scored code and full output
    if problem.latest_score != problem.best_score:
        source_code = problem.best_code
        save_to_disk(source_code, problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.best_score}_{current_time}.py", logger)
        full_output_file_best_score = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.best_score}_{current_time}.txt"
        
        output_best_score = await generate_output_async(source_code, problem.full_input_path, full_output_file_best_score, 45, logger)  # timeout in secs
        
        if output_best_score:
            logger.info(f'successful generated the best scored full output: {output_best_score}')

    return full_output_file

