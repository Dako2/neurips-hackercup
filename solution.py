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

class Solution:
    #input
    code: str 
    sample_input_path: Path
    sample_output_path: Path
    full_input_path: Path
    problem_name: str
    timestamp: str
    id: int

    #evaluation result
    score: float = Field(default=None) #[None, 0->1]
    sample_time_collapse: float = Field(default=100000.)
    sample_eval_status: str = Field(default='pending') # ['passed', 'failed', 'empty', 'error', 'timeout', 'pending']
    full_output_status: str = Field(default='pending') # ['complete', 'error', 'timeout', 'pending']
    sample_eval_report: str = Field(default='')
    #output
    code_path: Path = Field(default=None) #[None, "xxxx/xx.py"]
    full_output_path: Path = Field(default=None) #[None, "xxxx/xx.txt"]

    #sorting info
    model_capability: str = Field(default='gpt4') # GPT4, Claude, Gemini, etc.
    
    
    def __init__(self, code, problem_name, sample_input_path, sample_output_path, full_input_path, model_capability): #generating test report of sample data and full eval 
        
        self.solution_folder = "generated/"
        self.id = int(time.time())
        os.makedirs(self.solution_folder, exist_ok=True)
        self.timestamp = datetime.datetime.now().strftime("%y-%m-%d-%M-%f")
        self.problem_name = problem_name
        self.code = code
        self.sample_input_path = sample_input_path
        self.full_input_path = full_input_path
        self.sample_output_path = sample_output_path

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
        self.full_output_status = self.full_testreport.status
        self.model_capability = model_capability

        self.to_submit_signal = False
        
    def eval(self):
        self.testreport = evaluator_sample(self.code, self.sample_input_path, self.sample_output_path)
        self.sample_eval_report_path = self.solution_folder + self.problem_name + f'_{self.timestamp}_sample_eval.txt'
        with open(self.sample_eval_report_path, 'w') as outfile:
            outfile.write(self.testreport.message)

        self.score = round(self.testreport.success_rate_number, 2)
        self.sample_eval_report = self.testreport.message
        self.sample_time_collapse = self.testreport.time_collapse
        self.sample_eval_status = self.testreport.status
        
        if self.testreport.status in ["passed"]: #not in ["error", "timeout"]:
            self.full_output_path = self.solution_folder + self.problem_name + f'_{self.score}_{self.timestamp}_full_out.txt'
            self.full_testreport = generate_full(self.code, self.full_input_path, self.full_output_path)
            self.full_output_status = self.full_testreport.status
            if self.full_output_status in ["complete"]:
                self.to_submit_signal=True
        
        self.code_path = self.solution_folder + self.problem_name + f'_{self.score}_{self.timestamp}.py'
        with open(self.code_path, 'w') as f:
            f.write(self.code)        
        return self.testreport, self.full_testreport

    def gen_full(self):
        self.full_output_path = self.solution_folder + self.problem_name + f'_{self.score}_{self.timestamp}_full_out.txt'
        self.full_testreport = generate_full(self.code, self.full_input_path, self.full_output_path)
        self.full_output_status = self.full_testreport.status
        return self.full_output_path
    
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
            'eval_status': self.sample_eval_status, #success, fail
            'full_status': self.full_output_status, #success, fail
            'score': self.score,
            'code_path': self.code_path,
            'full_output_path': self.full_output_path,
            'model_capability': self.model_capability,
            'sample_time_collapse': self.sample_time_collapse,
            'sample_eval_report': self.sample_eval_report,
        }
    @property
    def key(self):
        return list(self.value.keys())

class SolutionManager:
    def __init__(self):
        self.solution_manager = pd.DataFrame()
        self.sol_dic = {}

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

        if bs['full_status'] in [None, 'error', 'timeout', 'pending']:
            full_output_path = sol.solution_folder + sol.problem_name + f'_{sol.score}_{sol.timestamp}_full_out.txt'
            full_testreport = generate_full(sol.code, sol.full_input_path, full_output_path, timeout=35)
 
        try:
            #sample_path = os.path.join(parent_folder, full_output_path+'.sample_eval')
            #with open(sample_path, 'w') as outfile:
            #    outfile.write(sol.sample_eval_report)
            shutil.copy2(bs['code_path'], os.path.join(parent_folder, Path(bs['code_path']).name))
            shutil.copy2(full_output_path, os.path.join(parent_folder, Path(bs['full_output_path']).name))

        except Exception as e:
            raise ValueError(f"Submission failed: {e}")
        
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

def generate_full(code, full_input_path, full_output_path, timeout=15):  # create new full output file for each case maybe
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
                if len(expected_output_cases) > 10:
                    message = "Generated output is not correct."
                else:
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
                message=f"error in execution of full evaluation",
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
