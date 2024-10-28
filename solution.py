from pathlib import Path
from pydantic import Field
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import datetime 
from lib.utils import TestReport, run_coroutine, save_to_disk, create_logger
import asyncio
import sys
import time
import traceback
import os 
import aiofiles
import shutil
import subprocess
import math

class SolutionManager:
    def __init__(self, exact_match = True):
        self.solution_manager = pd.DataFrame()
        self.solution_dict = {}

        self.exact_match = exact_match
        self.bs = None

    def add_solution(self, solution):
        self.solution_dict[solution.id] = solution

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
        self.bs = self.solution_manager.iloc[0] if not self.solution_manager.empty else None
        return self.bs
    
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
        sol = self.solution_dict[bs['id']] #solution class

        #full_output_path
        if sol.full_output_status not in ['complete'] or not Path(sol.full_gen_output_path).exists():
            _ = generate_full(sol.code, sol.full_input_path, sol.full_gen_output_path, timeout=90)
        
        new_path = Path("to_submit/") / Path(sol.full_gen_output_path).name.replace("score", str(round(sol.score, 2)))
        if Path(sol.full_gen_output_path).exists():
            shutil.copy2(sol.full_gen_output_path, new_path)
        new_path = Path("to_submit/") / Path(sol.sample_gen_output_path).name.replace("score", str(round(sol.score, 2)))
        if Path(sol.sample_gen_output_path).exists():
            shutil.copy2(sol.sample_gen_output_path, new_path)
        new_path = Path("to_submit/") / Path(sol.code_path).name.replace("score", str(round(sol.score, 2)))
        if Path(sol.code_path).exists():
            shutil.copy2(sol.code_path, new_path)

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

class Solution:
    def __init__(self, code, problem, model_capability, solver, lang="py"): #py, cpp
        self.code = code
        self.problem = problem
        self.model_capability = model_capability #GPT, Gemini, ...
        self.solver = solver #RAG, MCTS, BG, ...
        self.lang = lang

        self.to_submit_signal = False #indicator for submission / stop signal

        self.problem_name = problem.problem_name.replace(" ", "_")
        self.sample_input_path = problem.sample_input_path
        self.sample_exp_output_path = problem.sample_output_path
        self.full_input_path = problem.full_input_path

        self.exe_file = None
        
        #Generate paths to save in case
        self.id = int(time.time())
        self.solution_folder = "generated/"
        os.makedirs(self.solution_folder, exist_ok=True)

        self.timestamp = datetime.datetime.now().strftime("%y-%m-%d-%M-%f")
        self.full_gen_output_path = f'{self.solution_folder}/{self.problem_name}_{self.id}_full_out_score.txt'
        self.sample_gen_output_path = f'{self.solution_folder}/{self.problem_name}_{self.id}_sample_out_score.txt'
        self.code_path = f'{self.solution_folder}/{self.problem_name}_{self.id}_score.{self.lang}'
        
        self.score = 0.0 #[None, 0->1]
        self.q = 0.0 
        self.sample_eval_status = 'pending' # ['passed', 'failed', 'empty', 'error', 'timeout', 'pending']
        self.full_output_status = 'pending' # ['complete', 'error', 'timeout', 'pending']    
        self.sample_time_collapse = 1E9
        self.prompt = '' #json.dumps(messages)
        self.sample_eval_result = ''
    
        self.sample_testreport = TestReport(
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

    @property
    def to_submit_or_not(self,):
        return self.to_submit_signal
    
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
            'prompt': self.prompt,
            'code_path': self.code_path,
            'full_output_path': self.full_gen_output_path,
            'sample_output_path': self.sample_gen_output_path,
            'sample_time_collapse': self.sample_time_collapse,
            'sample_eval_result': self.sample_eval_result,
            'language': self.lang,
        }
    @property
    def key(self):
        return list(self.value.keys())
    
    def eval(self, exact_match=True, logger=None): #TODO:
        if not logger:
            logger = create_logger(f'logs/temp.log', f'temp')
        if self.lang == 'py':
            return self.eval_python(exact_match, logger)
        elif self.lang == 'cpp':
            return self.eval_cpp(exact_match, logger)
        else:
            raise NotImplementedError

    def eval_python(self, exact_match=True, logger=None): #TODO:
        self.sample_testreport = evaluator_sample(self.code, self.sample_input_path, self.sample_exp_output_path)
        save_to_disk(self.code, self.code_path)
        logger.info("sample evaluation done")
        save_to_disk(self.sample_testreport.message, self.sample_gen_output_path)

        self.score = round(self.sample_testreport.success_rate_number, 2)
        self.sample_eval_status = self.sample_testreport.status
        self.sample_eval_result = self.sample_testreport.message
        self.sample_time_collapse = self.sample_testreport.time_collapse
        
        if not exact_match or self.sample_eval_status in ['passed'] or self.score > 0.3:
            self.full_testreport = generate_full(self.code, self.full_input_path, self.full_output_path)
            self.full_output_status = self.full_testreport.status
            logger.info("full output gen done")
            if self.full_output_status in ["complete"] and self.sample_testreport.status in ['passed']:
                self.to_submit_signal=True
                logger.info("success! ready to submit!")
        return self.sample_testreport, self.full_testreport

    def eval_cpp(self, exact_match=True, logger=None): #TODO:
        compilation_res = compile_cpp(self.code, self.code_path)
        if not compilation_res.status == "success":
            self.full_testreport = compilation_res
            self.sample_testreport = compilation_res
            return self.sample_testreport, self.full_testreport
        
        self.exe_file = Path(self.code_path).with_suffix(".exe")
        logger.info("compilation done")
        
        self.sample_testreport = evaluator_sample_cpp(self.exe_file, self.sample_input_path, self.sample_exp_output_path)
        save_to_disk(self.sample_testreport.message, self.sample_gen_output_path)
        logger.info("sample evaluation done")
        
        self.score = round(self.sample_testreport.success_rate_number, 2)
        self.sample_eval_status = self.sample_testreport.status
        self.sample_eval_result = self.sample_testreport.message
        self.sample_time_collapse = self.sample_testreport.time_collapse
        
        if not exact_match or self.sample_eval_status in ['passed'] or self.score > 0.5:
            self.full_testreport = generate_full_cpp(self.exe_file, self.full_input_path, self.full_gen_output_path, timeout=120)
            self.full_output_status = self.full_testreport.status
            logger.info("full output gen done")
            if self.full_output_status in ["complete"] and self.sample_testreport.status in ['passed']:
                self.to_submit_signal=True
                logger.info("success! ready to submit!")
        return self.sample_testreport, self.full_testreport

    def erase_exe(self, exe_file):
        exe_file = Path(exe_file)
        try:
            if exe_file.exists():
                exe_file.unlink()
        except Exception as e:
            print(f"Failed to delete temporary file {exe_file}: {e}")

async def check_correctness(program: str, input_data: str, expected_output: str, timeout: float) -> TestReport:
    return await exec_program(program, input_data, expected_output, timeout)

def evaluator_sample(code, sample_input_path, sample_output_path):  # sample_test_report, code_path ==> heapq
    sample_input = Path(sample_input_path).read_text()
    sample_output = Path(sample_output_path).read_text()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_report = executor.submit(
            run_coroutine, 
            check_correctness(code, sample_input, sample_output, 3)
        )
        test_report_sample = future_report.result()
        return test_report_sample

def generate_full(code, full_input_path, full_output_path, timeout=35):  # create new full output file for each case maybe
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_report = executor.submit(
            run_coroutine, 
            generate_output_async(code, full_input_path, full_output_path, timeout)
        )
        test_report = future_report.result()
        return test_report

def generate_full_cpp(exe_file: Path, input_file: Path, output_file: Path, timeout: int = 30):  # create new full output file for each case maybe
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_report = executor.submit(
            run_coroutine, 
            generate_output_cpp_async(exe_file, input_file, output_file, timeout=timeout)
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
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_full=format(0, ".0%"),
                success_rate_number=0.0,
                failed_full=0,              
                status="error",
                message=f"error in execution of full evaluation",
                output=f"{output_file}",
                time_collapse=time.time()-starting_timer
            )

        # Write the stdout output to the output file
        await outfile.write(stdout.decode())

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
    
def compile_cpp(code: str, code_path: Path, timeout: int = 10) -> TestReport:
    """Compile C++ code and return a TestReport indicating the result."""
    cpp_file = Path(code_path)
    exe_file = Path(code_path).with_suffix('.exe')
    save_to_disk(code, cpp_file)  # Save the .cpp file

    compile_command = [
        'g++',
        '-std=c++17',
        '-O2',
        str(cpp_file),
        '-o',
        str(exe_file),
        '-Wl,-stack_size,0x20000000'
    ]
    try:
        subprocess.run(
            compile_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return TestReport(
            total=0,
            failed=0,
            success_rate="0%",
            success_rate_number=0.0,
            success_rate_full="0%",
            total_full=0,
            failed_full=0,
            status="success",
            message="Compilation successful",
            output="",
        )
    except subprocess.TimeoutExpired:
        return TestReport(
            total=0,
            failed=0,
            success_rate="0%",
            success_rate_number=0.0,
            success_rate_full="0%",
            total_full=0,
            failed_full=0,
            status="timeout",
            message=f"Compilation timed out after {timeout} seconds.",
            output="",
        )
    except subprocess.CalledProcessError as e:
        return TestReport(
            total=0,
            failed=0,
            success_rate="0%",
            success_rate_number=0.0,
            success_rate_full="0%",
            total_full=0,
            failed_full=0,
            status="error",
            message=f"Compilation failed: {e.stderr.decode().strip()}",
            output="",
        )


def evaluator_sample_cpp(exe_path, sample_input_path, sample_output_path):  # sample_test_report, code_path ==> heapq
    sample_input = Path(sample_input_path).read_text()
    sample_output = Path(sample_output_path).read_text()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_report = executor.submit(
            run_coroutine, 
            exec_program_cpp(exe_path, sample_input, sample_output, 15)
        )
        test_report_sample = future_report.result()
        return test_report_sample

# cpp: run the program code and compare the sample outputs TODO not being used
async def exec_program_cpp(exe_file, input_data: str, expected_output: str, timeout: float) -> TestReport:
    total = int(input_data.split("\n")[0])
    starting_timer = time.time()
    try:
        # Check if the executable file was created
        if not Path(exe_file).exists():
            return TestReport(
                total=total,
                failed=0,
                success_rate=format(0, ".0%"),
                success_rate_number=0.0,
                success_rate_full=format(0, ".0%"),
                total_full=total,
                failed_full=0,
                status="error",
                message=f"Compilation succeeded but executable not found: {str(exe_file)}",
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
    
async def generate_output_cpp_async(
    exe_file: Path, input_file: Path, output_file: Path, timeout: int
) -> TestReport:
    starting_timer = time.time()

    try:
        process = await asyncio.create_subprocess_exec(
            str(exe_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async with aiofiles.open(input_file, 'r') as infile:
            input_data = await infile.read()

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_data.encode()),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return TestReport(
                total=0,
                failed=0,
                success_rate="0%",
                success_rate_number=0.0,
                success_rate_full="0%",
                failed_full=0,
                status="timeout",
                message="Execution timed out.",
                output=str(output_file),
                time_collapse=time.time() - starting_timer,
            )

        return_code = await process.wait()
        if return_code != 0:
            return TestReport(
                total=0,
                failed=0,
                success_rate="0%",
                success_rate_number=0.0,
                success_rate_full="0%",
                failed_full=0,
                status="error",
                message=f"Execution failed with return code {stderr}.",
                output=str(output_file),
                time_collapse=time.time() - starting_timer,
            )

        async with aiofiles.open(output_file, 'w') as outfile:
            await outfile.write(stdout.decode())

        return TestReport(
            total=0,
            failed=0,
            success_rate="0%",
            success_rate_number=0.0,
            success_rate_full="0%",
            failed_full=0,
            status="complete",
            message=f"Output written to {output_file}",
            output=str(output_file),
            time_collapse=time.time() - starting_timer,
        )

    except Exception as e:
        return TestReport(
            total=0,
            failed=0,
            success_rate="0%",
            success_rate_number=0.0,
            success_rate_full="0%",
            failed_full=0,
            status="error",
            message=f"An error occurred: {str(e)}",
            output=str(output_file),
            time_collapse=time.time() - starting_timer,
        )
    
# if tempreport.status == 'complete': check_output()
# ad-hoc purpose to check generated_solution txt vs txt/out
def check_output(input_data_file, model_output_file, expected_output_file):
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
        status='complete',
        message=f"the full report has {success_rate_number} correction ratio.", 
        output="",
    )


if __name__ == '__main__':

    from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training, load_problem_v2024
    from pathlib import Path
    from lib.utils import (
        create_logger,
        load_problem_from_folder,
        verify_code_syntax,
        extract_text,
        maybe_remove_backticks,
        save_to_disk,)
    
    SELECT_LANGUAGE = 'cpp'
    logger=create_logger("temp.log","temp")
    
    problem_directory = "dataset/2024/Round2"
    problem_names = list_problem_names(problem_directory, "2024")
    problem_list = []
    for problem_name in problem_names:
        problem_list.append(load_problem_v2024(problem_name, Path(problem_directory)))
    problem = problem_list[1]

    print(problem.problem_name)

    from lib.llms import LLM
    strong_llm = LLM(model_name="o1")
    fast_llm = LLM(model_name="gpt4")
    simple_initial_advisor_prompt = """Write a complete code program to solve complex algorithmic problems. 
    - Ensure that the code includes a main function and is structured so that the program is executable when run, following the conventions of {selected_language}.
    - Do not use any external libraries; only stick to {selected_language} standard library
    Hint: Highly optimized and suited for large inputs, using advanced techniques like Fenwick Trees and binary search but is more complex to understand and debug.
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

        code = fast_llm.run_messages(messages=messages1)
        if '<source_code>' in code:
            code = extract_text(code, '<source_code>')

        code = maybe_remove_backticks(code)
        return code
        
    #code = Path("generated//Bunny_Hopscotch_1730085454_score.cpp").read_text()
    #code = Path("/Users/dako22/Downloads/bunny_hopscotch__molamola_source_code.txt").read_text() + '\n'
    
    #code = manager(problem)
    #code = worker(code, SELECT_LANGUAGE)

    code = Path("test_code.cpp").read_text()
    print(code)
    s = Solution(code, problem, "gpt4", "zeroshot", lang="cpp") #py, cpp
    testreport, fullreport = s.eval(logger=logger)
    sm=SolutionManager()
    sm.add_solution(s)
    sm.to_submit("to_submit/")


    