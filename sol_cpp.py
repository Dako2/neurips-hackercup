from pathlib import Path
from pydantic import Field
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import datetime 
from lib.utils import TestReport, run_coroutine
import asyncio
import sys
import time
import traceback
import os 
import aiofiles
import shutil
import math
import subprocess
from lib.utils import (
    create_logger,
    load_problem_from_folder,
    verify_code_syntax,
    extract_text,
    maybe_remove_backticks,
    save_to_disk,
)
from logging import Logger

logger = create_logger(f'logs/mcts_cpp_.log', f'gpt4')

SELECT_LANGUAGE = "cpp"

def save_to_disk(content: str, path: Path,):
    path.parent.mkdir(parents=True, exist_ok=True)
    print(content)
    with path.open("w", encoding ="utf-8") as f:
        f.write(content)

class Solution:
    #input
    code: str 
    sample_input_path: Path
    sample_output_path: Path
    full_input_path: Path
    problem_name: str
    timestamp: str
    id: int
    
    #output
    code_path: Path = Field(default=None) #[None, "xxxx/xx.py"]
    #evaluation result
    score: float = Field(default=None) #[None, 0->1]
    q: float = Field(default=None)
    prompt: float = Field(default=None)

    sample_time_collapse: float = Field(default=100000.)
    
    sample_eval_status: str = Field(default='pending') # ['passed', 'failed', 'empty', 'error', 'timeout', 'pending']
    full_output_status: str = Field(default='pending') # ['complete', 'error', 'timeout', 'pending']
    
    sample_eval_report: str = Field(default='')
    #sorting info
    model_capability: str = Field(default='gpt4') # GPT4, Claude, Gemini, etc.
    solver: str = Field(default='NaN')
    
    def __init__(self, code, problem, problem_name, sample_input_path, sample_output_path, full_input_path, model_capability, logger=None): #generating test report of sample data and full eval 
        
        self.solution_folder = "generated/"
        self.id = int(time.time())
        os.makedirs(self.solution_folder, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%y-%m-%d-%M-%f")
        self.problem_name = problem_name
        self.code = code
        self.problem = problem
        self.sample_input_path = sample_input_path
        self.full_input_path = full_input_path
        self.sample_output_path = sample_output_path
        self.full_output_path = None        

        self.testreport = None
        self.full_testreport = TestReport(
            total=0,
            failed=0,
            success_rate=format(0, ".0%"),
            success_rate_number=0.0,
            success_rate_full=format(0, ".0%"),
            failed_full=0,    
            status="pending", 
            message=f"not evaluated",
            output=f"",
            time_collapse=0
        )
        self.full_output_status = None
        self.model_capability = model_capability
        self.to_submit_signal = False
        self.logger = logger
        
    def eval(self, exact_match = True, logger=None): #TODO:
        
        self.testreport = self.evaluator_sample(self.code, self.sample_input_path, self.sample_output_path)
        self.score = round(self.testreport.success_rate_number, 2)

        self.sample_eval_report_path = self.solution_folder + self.problem_name + f'_{self.score}_{self.timestamp}_sample_eval.txt'
        with open(self.sample_eval_report_path, 'w') as outfile:
            outfile.write(self.testreport.message)

        self.sample_eval_report = self.testreport.message
        self.sample_time_collapse = self.testreport.time_collapse
        self.sample_eval_status = self.testreport.status
        
        if not exact_match or self.testreport.status in ['passed'] or self.testreport.success_rate_number > 0.3:
            self.full_output_path = self.solution_folder + self.problem_name + f'_{self.score}_{self.timestamp}_full_out.txt'
            self.full_testreport = self.generate_full_cpp(self.code, SELECT_LANGUAGE, logger)
            self.full_output_status = self.full_testreport.status
            if self.full_output_status in ["complete"] and self.testreport.status in ['passed']:
                self.to_submit_signal=True
        
        self.code_path = self.solution_folder + self.problem_name + f'_{self.score}_{self.timestamp}.py'
        with open(self.code_path, 'w') as f:
            f.write(self.code)        
        return self.testreport, self.full_testreport
     
    def generate_sample_cpp(self, code, selected_language, sample_output): # DAKO 2nd edit
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_report = executor.submit(run_coroutine, write_sample_output_cpp(1, code, self.problem, sample_output, selected_language, timeout=30, logger=self.logger))
            sample_output_file = future_report.result()
            self.logger.info(f"write the sample out to: {sample_output_file}")
            return sample_output_file

    # DAKO: generate full output here for cpp 
    def generate_full_cpp(self, code, selected_language, logger):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_report = executor.submit(run_coroutine, write_full_output(1, code, self.problem, selected_language, logger))
            full_output_file = future_report.result()
            self.logger.info(f"write the full out to: {sample_output_file}")
            return full_output_file  
    
    @property
    def check(self):
        if self.testreport and self.full_testreport:
            return True
        else:
            return False
        
    @property
    def value(self):
        return {
            'id': self.id,
            'score': self.score,
            'q': self.q, 
            'eval_status': self.sample_eval_status, #success, fail
            'full_status': self.full_output_status, #success, fail
            'model_capability': self.model_capability,
            'solver': self.solver,
            'problem_name': self.problem_name,
            'code': self.code,
            'code_path': self.code_path,
            'full_output_path': self.full_output_path,
            'sample_time_collapse': self.sample_time_collapse,
            'prompt': self.prompt,
            'sample_eval_report': self.sample_eval_report,
        }
    @property
    def key(self):
        return list(self.value.keys())
    
    # DAKO: evaluator here: compile and exe .cpp, write txt to sample_output_file, compare txt vs. sample_output_solution and create test report
    def evaluator_sample(self, code, sample_input_path, sample_output_path):  # sample_test_report, code_path ==> heapq
        sample_input = Path(sample_input_path).read_text()
        sample_output = Path(sample_output_path).read_text()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            if SELECT_LANGUAGE == "python":   
                future_report = executor.submit(
                    run_coroutine, 
                    check_correctness(code, sample_input, sample_output, 3) # exe, compare txt
                )
                test_report_sample = future_report.result()
            
            elif SELECT_LANGUAGE == "cpp": # DAKO 1st edit
                sample_output_file = self.generate_sample_cpp(code, SELECT_LANGUAGE, sample_output) # compiple and execute cpp -> sample_output txt
                test_report_sample = self.check_output(sample_input_path, sample_output_file, sample_output_path)
            
            return test_report_sample
        


    # ad-hoc purpose to check generated_solution txt vs txt/out
    def check_output(self, input_data_file, model_output_file, expected_output_file):
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
                time_collapse=0,
            )
    
        is_number_tag = is_number(expected_output_cases[0].split(":")[1].strip())

        for i in range(len(actual_output_cases)):
            # if not empty "" which can't do split(":")
            if actual_output_cases[i].strip() and expected_output_cases[i].strip():
            # if both numbers, compare up to 1e-7 decimals
                if is_number_tag:
                    expected_value = float(expected_output_cases[i].split(":")[-1].strip()) # "Case #1: 20.710678118654748" -> 20.71067811865474
                    actual_value = float(actual_output_cases[i].split(":")[-1].strip())
                    if math.isclose(expected_value, actual_value, abs_tol=1e-7):
                        success += 1
                    else:
                        failed += 1
                # if not numbers, perform a direct string comparison
                else:
                    if (expected_output_cases[i] == actual_output_cases[i]):
                        success = success + 1
                    else:
                        failed = failed + 1

        success_rate = format(success/total, ".0%")
        success_rate_number = float(success/total)
        if success_rate_number < 1:
            return TestReport(
                status='FAILED',
                message=f"The sample test has a success rate of {success_rate}", #message,
                total=total,
                failed= failed,
                success_rate = success_rate,
                success_rate_full = success_rate,
                success_rate_number = success_rate_number,
                total_full=total,
                failed_full=failed,    
                output="", #f"{stdout.decode()}",
                time_collapse=0,
            )
        else:
            return TestReport(
                status='passed',
                message="The sample evaluation succeeds.", #message,
                total=total,
                failed= failed,
                success_rate = success_rate,
                success_rate_full = success_rate,
                success_rate_number = success_rate_number,
                total_full=total,
                failed_full=failed,    
                output="", #f"{stdout.decode()}",
                time_collapse=0,
            )

def is_number(s):
    try:
        float(s)  # Attempt to convert the string to a float
        return True
    except ValueError:
        return False


class SolutionManager:
    def __init__(self, exact_match = True):
        self.solution_manager = pd.DataFrame()
        self.sol_dic = {}
        self.exact_match = exact_match

    def add_solution(self, solution):
        self.sol_dic[solution.id] = solution
        new_solution_df = pd.DataFrame([solution.value])
        self.solution_manager = pd.concat(
            [self.solution_manager, new_solution_df], 
            ignore_index=True
        )

    """
    "eval_status": Sorting by ['passed', 'failed', 'empty', 'error', 'timeout', 'pending']
    "full_status": Sorting by ['success', 'error', 'timeout', 'pending']
    "model_order": Sorting by ['o1', 'gpt4', 'claude', 'gemini', 'gpt3.5']
    "score" ... 
    "model_capability" ... 
    "sample_time_collapse" ... 
    """
    def sort_solutions(self):
        # Define the custom order for sample_eval_status
        eval_status_order = ['passed', 'failed', 'empty', 'error', 'timeout', 'pending']
        self.solution_manager['eval_status'] = pd.Categorical(
            self.solution_manager['eval_status'], 
            categories=eval_status_order, 
            ordered=True
        )
        
        # Define the custom order for full_output_status
        output_status_order = ['complete', 'error', 'timeout', 'pending']
        self.solution_manager['full_status'] = pd.Categorical(
            self.solution_manager['full_status'], 
            categories=output_status_order, 
            ordered=True
        )
        
        # Define the custom order for model_capability
        model_order = ['o1', 'o1-mini','gpt4', 'claude', 'gemini', 'gpt3.5']
        self.solution_manager['model_capability'] = pd.Categorical(
            self.solution_manager['model_capability'], 
            categories=model_order, 
            ordered=True
        )
        
        # Sorting by multiple columns with custom sorting orders
        self.solution_manager = self.solution_manager.sort_values(
            by=['eval_status', 'full_status', 'score', 'model_capability', 'sample_time_collapse'],
            ascending=[True, True, False, True, True]  # Adjust ascending order as needed
        ).reset_index(drop=True)

    def get_top_solutions(self, top_n=3):
        self.sort_solutions()
        if self.solution_manager.empty:
            return pd.DataFrame()  # Return an empty DataFrame if no solutions are available
        else:
            return self.solution_manager.head(min(top_n, len(self.solution_manager)))

    def best_solution(self):
        self.sort_solutions()
        return self.solution_manager.iloc[0] if not self.solution_manager.empty else None
    
    def to_submit(self, parent_folder):
        """
        TBD: implement algorithms
        if deterministic:
           sol = self.best_solution()
        else:
           candidates = self.get_top_solutions(self, top_n=3)
           #call LLM to compare candidates and make decision
        """
        os.makedirs(parent_folder, exist_ok=True)
        
        bs = self.best_solution() #pd row
        sol = self.sol_dic[bs['id']] #solution class
        full_output_path = bs['full_output_path']

        sample_path = parent_folder + '/' + sol.problem_name + f'_{sol.score}_{sol.timestamp}_sample_eval.txt'
        with open(Path(sample_path), 'w') as outfile:
            outfile.write(sol.sample_eval_report)
        code_path = parent_folder + '/' + sol.problem_name + f'_{sol.score}_{sol.timestamp}.py'
        with open(Path(code_path), 'w') as outfile:
            outfile.write(sol.code)
        
        #full_output_path
        full_out_path = parent_folder + '/' + sol.problem_name + f'_{sol.score}_{sol.timestamp}_full_out.txt' 
        if sol.full_output_status not in ['complete'] or not sol.full_output_path:
            _ = generate_full(sol.code, sol.full_input_path, full_out_path, timeout=35)
        else:
            shutil.copy2(sol.full_output_path, full_out_path)

    def save(self, file_path = "solution_manager.pickle"):
        # Step 1: Load the existing pickle file
        try:
            existing_df = pd.read_pickle(file_path)
        except FileNotFoundError:
            # If the file doesn't exist, create an empty DataFrame
            existing_df = pd.DataFrame()
        # Append the new dataframe to the existing one
        combined_df = pd.concat([existing_df, self.solution_manager], ignore_index=True)
        # Step 3: Save the combined dataframe back to the pickle file
        combined_df.to_pickle(file_path)


async def check_correctness(program: str, input_data: str, expected_output: str, timeout: float) -> TestReport:
    return await exec_program(program, input_data, expected_output, timeout)
        

def generate_full(code, full_input_path, full_output_path, timeout=35):  # create new full output file for each case maybe
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_report = executor.submit(
            run_coroutine, 
            generate_output_async(code, full_input_path, full_output_path, timeout)
        )
        test_report = future_report.result()
        return test_report
    
async def exec_program(program, input_data, expected_output, timeout):
    total = input_data.split("\n")[0]
    starting_timer = time.time()
    if not program:
        return TestReport(
            total=total,
            failed=0,
            success_rate=format(0, ".0%"),
            success_rate_number=0.0,
            success_rate_full=format(0, ".0%"),
            failed_full=0,    
            status="error", 
            message=f"the source code is empty",
            output=f"",
            time_collapse=0
        )
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", program,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_data.encode()), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.terminate()
            await process.wait()  # Wait for the process to finish
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_number=0.0,
                success_rate_full=format(0, ".0%"),
                failed_full=0,             
                status="timeout",
                message=f"Took too long! Your program timed out after {timeout} seconds of execution.",
                output=f"",
                time_collapse=timeout
            )
        
        if process.returncode != 0:
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_full=format(0, ".0%"),
                success_rate_number=0.0,
                failed_full=0,    
                status="error", 
                message=f"Program execution failed: {stderr.decode()}",
                output=f"",
                time_collapse=time.time()-starting_timer
            )
        else:
            if stdout.decode().strip() == expected_output.strip():
                return TestReport(
                    total=total,
                    failed=0,
                    success_rate=format(1, ".0%"),
                    success_rate_number=1.0,
                    success_rate_full=format(1, ".0%"),
                    failed_full=0,    
                    status='passed', 
                    message=f"The program successfully passed {total} sample results with a time consumption of {time.time()-starting_timer}ms",
                    output=f"{stdout.decode()}",
                    time_collapse=time.time()-starting_timer
                )
            else:
                actual_output = stdout.decode()
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
                        failed_full=0,              
                        status="empty",
                        message=f"There is no output generated from the source code.",
                        output=f"",
                        time_collapse=time.time()-starting_timer
                    )
                for i in range(len(expected_output_cases)):
                    if i >= len(actual_output_cases):
                        failed += 1
                        continue
                    if expected_output_cases[i] == actual_output_cases[i]:
                        success += 1
                    else:
                        failed += 1
                success_rate = format(success / len(expected_output_cases), ".0%")
                success_rate_number = float(success / len(expected_output_cases))
                 
                message = f"<expected>\n{expected_output}</expected>\n---\n<got>\n{stdout.decode()}</got>"
                
                return TestReport(
                    total=total,
                    failed=failed,
                    success_rate=success_rate,
                    success_rate_full=success_rate,
                    success_rate_number=success_rate_number,
                    failed_full=failed,    
                    status='failed',
                    message=message,
                    output=f"{stdout.decode()}",
                    time_collapse=time.time()-starting_timer
                )
    except Exception:
        return TestReport(
            total=total,
            failed=0,
            success_rate=format(0, ".0%"),
            success_rate_full=format(0, ".0%"),
            success_rate_number=0.0,
            failed_full=0,              
            status="error",
            message=f"An error occurred: {traceback.format_exc()}",
            output=f"",
            time_collapse=0.0
        )

async def generate_output_async(program: str, input_file: Path, output_file: Path, timeout: int):
    starting_timer = time.time()
    # Open the input file for reading and output file for writing
    async with aiofiles.open(input_file, 'r') as infile, aiofiles.open(output_file, 'w') as outfile:
        # Read the input file content
        input_data = await infile.read()
        total = input_data.split("\n")[0]
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
            process.terminate()  # More graceful than kill
            await process.wait()  # Wait for the process to finish
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_full=format(0, ".0%"),
                success_rate_number=0.0,
                failed_full=0,              
                status="timeout",
                message=f"!!!!!!Time OUT for executing {total} cases",
                output=f"{output_file}",
                time_collapse=time.time()-starting_timer
            )
        # Check the return code after the process completes
        return_code = process.returncode
        if return_code != 0:
            #logger.warning(f"Process exited with non-zero return code {return_code}")
            #logger.error(f"Error: {stderr.decode().strip()}")
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_full=format(0, ".0%"),
                success_rate_number=0.0,
                failed_full=0,              
                status="error",
                message=f"Error: {stderr.decode().strip()}",
                output=f"{output_file}",
                time_collapse=time.time()-starting_timer
            )

        # Write the stdout output to the output file
        await outfile.write(stdout.decode())
        #logger.info(f"Process completed successfully. Output written to {output_file}")

        return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_full=format(0, ".0%"),
                success_rate_number=0.0,
                failed_full=0,              
                status="complete",
                message=f"Complete the full evaluation. Output written to {output_file}",
                output=f"{output_file}",
                time_collapse=time.time()-starting_timer
            )


# saved both .py/.cpp and .txt file to generated folder
async def write_sample_output_cpp(stage, source_code, problem, sample_output, selected_language, timeout = 30, logger = None): 
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    sample_output_file = "generated/" + f"{problem.problem_name.replace(' ', '_')}_{selected_language}_{current_time}.txt"
    #sample_output_file = sample_output
    output = await generate_output_cpp_async(current_time, problem, source_code, problem.sample_input_path, sample_output_file, timeout, logger)  # timeout in secs
    if output:
        logger.info(f'successful generated the most recent sample output: {output}')

    return sample_output_file

async def generate_output_cpp_async(current_time, problem, program: str, input_file: Path, output_file: Path, timeout: int, logger: Logger):
    # current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    exe_file = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}__cpp_{current_time}.exe"
    cpp_file = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_0_cpp_{current_time}.cpp"
    save_to_disk(program, cpp_file) # save the .cpp file as we save the output. this is the last code

    # should create and save the file .exe
    # compile_command = f'g++ -std=c++17 -O2 "{cpp_file}" -o "{exe_file}"' 
    compile_command = f'g++ -std=c++17 -O2 "{cpp_file}" -o "{exe_file}" -Wl,-stack_size,0x20000000' # 20m for 512MB/40m for 1G for stack overflow
    # into below to capture the error messages
    try:
        result = subprocess.run(
            compile_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Compilation succeeded: {result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Compilation failed: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to compile C++ code: {e.stderr.decode()}")

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

    # except subprocess.TimeoutExpired as e:
    # except subprocess.CalledProcessError as e:

    except Exception as e:
        logger.error(f"An unexpected error occurred during the execution of {problem_name}: {str(e)}")
        raise e

    return str(output_file) # return the path to the output file for generated results

# DAKO for both python and cpp, but muting python here as you can use your own program
async def write_full_output(stage, source_code, problem, selected_language, logger = None): 
    current_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    full_output_file = problem.submit_folder / f"{problem.problem_name.replace(' ', '_')}_{problem.latest_score:.1f}_{selected_language}_{current_time}.txt"
    
    # try:
    # if selected_language == "python": # below function saved .py/.cpp and output .txt
    #     output = await generate_output_async(current_time, problem, source_code, problem.full_input_path, full_output_file, 30, logger)  # timeout in secs
    # elif selected_language == "cpp":
    output = await generate_output_cpp_async(current_time, problem, source_code, problem.full_input_path, full_output_file, 90, logger)  # timeout in secs
    
    if output:
        logger.info(f'successful generated the most recent full output: {output}')

    return full_output_file

if __name__ == '__main__':

    from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training, load_problem_v2024
    from pathlib import Path

    problem_directory = "dataset/2024/Round2"
    problem_names = list_problem_names(problem_directory, "2024")
    problem_list = []
    for problem_name in problem_names:
        problem_list.append(load_problem_v2024(problem_name, Path(problem_directory)))
    problem = problem_list[1]

    print(problem.problem_name)

    from lib.llms import LLM
    strong_llm = LLM(model_name="gpt4")
    simple_initial_advisor_prompt = """Write a complete code program to solve complex algorithmic problems. 
    - Ensure that the code includes a main function and is structured so that the program is executable when run, following the conventions of {selected_language}.
    - Do not use any external libraries; only stick to {selected_language} standard library

    Problem: {problem}"""

    def manager(problem): #implement the code; fixed context length simple response
        """Processes assistant output to extract and verify the source code."""
        messages = [{'role': 'user', 'content': simple_initial_advisor_prompt.format(problem=problem.problem_description, selected_language="cpp")}]
        out = strong_llm.run_messages(messages=messages)
        
        code = extract_text(out, '<source_code>')
        code = maybe_remove_backticks(code)
        return code
    
    # Using
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

    # worker to extract code
    def worker(out0, selected_language):
        extract_code_message = EXTRACT_CODE_PROMPT.format(
            output=out0,
            selected_language=selected_language)
        
        messages1 = [
            {
                'role': 'user',
                'content': extract_code_message
            }
        ]

        code = strong_llm.run_messages(messages=messages1)
        if '<source_code>' in code:
            code = extract_text(code, '<source_code>')

        code = maybe_remove_backticks(code)
        return code
        
    code = manager(problem)
    code = worker(code, SELECT_LANGUAGE)
    print(code)

    s = Solution(code, problem, problem.problem_name, problem.sample_input_path, problem.sample_output_path, problem.full_input_path, "gpt4", logger) #generating test report of sample data and full eval 
    testreport, fullreport = s.eval(logger=logger)

