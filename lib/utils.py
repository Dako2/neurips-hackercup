import asyncio
import subprocess
import datetime
import json
import logging
import multiprocessing
import os
import pathlib
import re
import sys
import traceback
from typing import Any, List, Optional

from pydantic import BaseModel, Field
from tree_sitter_languages import get_language, get_parser
 
from logging import Logger
from pathlib import Path
import pyzipper
import warnings
warnings.simplefilter('ignore', FutureWarning)

filtered_ds = None
problem_index = 0

language = get_language("python")
tree_parser = get_parser("python")

def verify_code_syntax(code_str):
    try:
        compile(code_str, '<string>', 'exec')
        return True
    except SyntaxError as e:
        return False
    
def extract_text(input_string, format):
    # Use a regex pattern to extract text between <prompt> and </prompt>
    #match = re.search(f'{format}(.*?){format.replace('<','</')}', input_string, re.DOTALL)
    match = re.search(f'{format}(.*?){format.replace("<", "</")}', input_string, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return input_string
    
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

# Run coroutine in a thread-safe manner
def run_coroutine(coro):
    try:
        # Check if the thread has an event loop, if not, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def load_problem_from_folder(year: str, unzipped_path: str, problem_name: str, logger: Logger):
    # Initialization
    if year == '2024':
        logger.info("Starts to load problem.")
        problem = load_problem_v2024(problem_name, Path(unzipped_path))
        logger.info(f"Finishes to load problem -- {problem.problem_name}.")
    else:
        logger.info("Starts to load problem.")
        problem = load_problem(problem_name, Path(unzipped_path))
        logger.info(f"Finishes to load problem -- {problem.problem_name}.")

    return problem

def unzip_questions(zip_path, password, unzipped_path):
    os.makedirs(unzipped_path, exist_ok=True)
    try:
        with pyzipper.AESZipFile(zip_path, 'r') as zip_ref:
            zip_ref.setpassword(password.encode('utf-8'))
            zip_ref.extractall(unzipped_path)
    except RuntimeError as e:
        if "Bad password" in str(e):
            raise ValueError("The password provided is incorrect.")
        else:
            raise ValueError(f"Runtime error: {str(e)}")
    except pyzipper.BadZipFile:
        raise ValueError("The ZIP file is corrupted or invalid.")
    
    subfolders = [f.name for f in os.scandir(unzipped_path) if f.is_dir()]
    print("Subfolders extracted:", subfolders)
    return subfolders
 
def create_logger(log_file, logger_name):
    # Create a logger with a unique name (logger_name should be unique per task)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Check if a FileHandler already exists
    file_handler_exists = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)
    stream_handler_exists = any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)

    if file_handler_exists and stream_handler_exists:
        return logger

    # Create a file handler for logging to a specific file
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        # Formatter for the logs
        formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
 
    if not file_handler_exists:
        # Create a file handler for logging to a specific file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter for the logs
        formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the new handler to the logger
        logger.addHandler(file_handler)

    # If no StreamHandler exists, create one
    if not stream_handler_exists:
        # Create a stream handler for logging to the terminal
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # Formatter for the logs (same as for file, can be customized separately)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    # Prevent propagation to the root logger
    logger.propagate = False

    return logger


def set_problem_index(index):
    global problem_index
    problem_index = index 

def remove_extra_newlines(text: str) -> str:
    # Use regex to replace 2 or more newlines (with possible whitespace in between) with a single newline
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text

def remove_comments_and_docstrings(code):
    # Define queries to capture comments and docstrings
    doc_str_pattern = """
    (module . (expression_statement (string)) @module_doc_str)
    (class_definition body: (block . (expression_statement (string)) @class_doc_str))
    (function_definition body: (block . (expression_statement (string)) @function_doc_str))
    """

    comment_pattern = "(comment) @comment"
    # Parse the code
    tree = tree_parser.parse(code.encode())
    root_node = tree.root_node

    # Query the tree for docstrings and comments
    doc_str_query = language.query(doc_str_pattern)
    doc_strs = doc_str_query.captures(root_node)

    comment_query = language.query(comment_pattern)
    comments = comment_query.captures(root_node)

    # Get the start and end points of all docstrings and comments
    doc_str_points = set((node.start_byte, node.end_byte) for node, _ in doc_strs)
    comment_points = set((node.start_byte, node.end_byte) for node, _ in comments)

    # Create a set of all points to remove
    remove_points = doc_str_points.union(comment_points)

    # Reconstruct the code, skipping over the parts to remove
    cleaned_code = []
    last_index = 0
    for start, end in sorted(remove_points):
        if last_index < start:
            cleaned_code.append(code[last_index:start])
        last_index = end

    # Add any remaining code after the last comment/docstring
    cleaned_code.append(code[last_index:])

    return "".join(cleaned_code)


def clean_code_string(code: str) -> str:
    code = remove_comments_and_docstrings(code)
    code = remove_extra_newlines(code)
    return code

class TestReport(BaseModel):
    status: str
    message: str
    success_rate: str
    success_rate_number: float
    total: int
    failed: int
    success_rate_full: str
    failed_full: int
    output: str
    time_collapse: float

    @property
    def content(self) -> str:
        # Create a formatted output string with all attributes
        output = (
            f"Status: {self.status}\n"
            f"Message: {self.message}\n"
            f"Success Rate: {self.success_rate}\n"
            f"Total: {self.total}\n"
            f"Failed: {self.failed}\n"
            f"Success Rate Full: {self.success_rate_full}\n"
            f"Failed Full: {self.failed_full}\n"
            f"Time Collapse: {self.time_collapse}\n"
        )
        return output

    @property
    def as_xml(self) -> str:
        return f"""
<test_report>
<status>{self.status}</status>
<message>{self.message}</message>
<output_file>{self.output}</output_file>
</test_report>
"""

# Helper functions for extracting parts of the text
def extract_block(text: str, start_marker: str, end_marker: str = None) -> str:
    """Extracts a block of text between two markers (or until the end of the text)."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return ""
    
    start_idx += len(start_marker)
    if end_marker:
        end_idx = text.find(end_marker, start_idx)
        return text[start_idx:end_idx].strip() if end_idx != -1 else text[start_idx:].strip()
    return text[start_idx:].strip()

def extract_list(text: str, start_marker: str, end_marker: str = None) -> List[str]:
    """Extracts a list of strings from text between two markers."""
    block = extract_block(text, start_marker, end_marker)
    if (start_marker == "<keywords>"):
        return [item.strip() for item in block.split(",") if item.strip()]
    else:
        return [item.strip() for item in block.split("\n") if item.strip()]

# Main function to format the response based on the model type
def format_response_python(text: str, model: Any) -> Any:
    """Format the response based on the model class."""
    if model == Analysis:
        core_question = extract_block(text, "<core_question>", "</core_question>")
        problem_solving_info = extract_list(text, "<problem_solving_info>", "</problem_solving_info>")
        algorithm = extract_block(text, "<algorithm>", "</algorithm>")
        tutorial = extract_block(text, "<tutorial>", "</tutorial>")
        plan = extract_block(text, "<plan>", "</plan>")
        pseudocode = extract_block(text, "<pseudocode>", "</pseudocode>")
        
        # Return an instance of Analysis populated with extracted data
        return Analysis(
            core_question=core_question,
            problem_solving_info=problem_solving_info,
            algorithm=algorithm,
            tutorial=tutorial,
            plan=plan,
            pseudocode=pseudocode
        )
    
    elif model == Solution:
        core_question = extract_block(text, "<core_question>", "</core_question>")
        problem_solving_info = extract_list(text, "<problem_solving_info>", "</problem_solving_info>")
        algorithm = extract_block(text, "<algorithm>", "</algorithm>")
        tutorial = extract_block(text, "<tutorial>", "</tutorial>")
        plan = extract_block(text, "<plan>", "</plan>")
        pseudocode = extract_block(text, "<pseudocode>", "</pseudocode>")
        source_code = extract_block(text, "<source_code>", "</source_code>")

        return Solution(
            core_question=core_question,
            problem_solving_info=problem_solving_info,
            algorithm=algorithm,
            tutorial=tutorial,
            plan=plan,
            pseudocode=pseudocode,
            source_code=source_code
        )

    elif model == Reflection:
        reflection = extract_block(text, "<reflection>", "</reflection>")
        keywords = extract_list(text, "<keywords>", "</keywords>")
        step_by_step_solution = extract_block(text, "<step_by_step_solution>", "</step_by_step_solution>")
        instructions = extract_list(text, "<instructions>", "</instructions>")
        general_advice = extract_list(text, "<general_advice>", "</general_advice>")

        return Reflection(
            reflection=reflection,
            keywords=keywords,
            step_by_step_solution=step_by_step_solution,
            instructions=instructions,
            general_advice=general_advice
        )     
    
    # If model type does not match, return None
    return None

class Problem(BaseModel):
    problem_dir: Optional[pathlib.Path] = Field(
        default=None, description="The path to the problem directory"
    )
    outputformat_match_or_not: str = Field(default=None, description="deterministic or not")
    problem_name: str = Field(..., description="The name of the problem")
    problem_description: str = Field(..., description="The description of the problem")
    
    sample_input: str = Field(..., description="The sample input of the problem")
    sample_output: str = Field(..., description="The sample output of the problem")
    full_input: Optional[str] = Field(default=None, description="The input of the problem", nullable=True)
    full_output: Optional[str] = Field(default=None, description="The output of the problem", nullable=True)
    
    full_input_path: Path = Field(default=None, description="The path to the input file")
    sample_input_path: Path = Field(default=None, description="The sample output path of the problem")
    sample_output_path: Path = Field(default=None, description="The sample output path of the problem")
    full_output_path: Path = Field(default=None, description="The path to the output file") #regulate the output file path; tbd
    
    problem_input: Optional[pathlib.Path] = Field(default=None, description="The path to the input file")
    problem_output: Optional[pathlib.Path] = Field(default=None, description="The path to the output file")
    
    best_code: str = Field(default="", description="best code")
    best_score: int = Field(default=0, description="best score")

    starting_time: int = Field(None, description="started epoch time in integer") # to initiate as None
    
    submit_folder: Path = Field(default=None, description="The path to the output file") #regulate the output file path; tbd
    @property
    def as_xml(self) -> str:
        return f"""
<problem>
<problem_statement>
{remove_extra_newlines(self.problem_description)}
</problem_statement>
<sample_test_cases>
<sample_input>
{self.sample_input}
</sample_input>
<sample_output>
{self.sample_output}
</sample_output>
</sample_test_cases>
</problem>
"""
    
# load problem for hacker cup 2024 format
def load_problem_v2024(problem_name: str, problem_dir: pathlib.Path) -> Problem:
    problem_subdir = problem_dir / f"{problem_name}" 
    problem_input = problem_subdir / "full_in.txt" #f"{problem_name.lower().replace(' ','_')}.in"
    problem_output = problem_subdir / f"{problem_name.lower().replace(' ','_')}.out"#f"{problem_name}_output" / "full_out.txt"
    sample_input = problem_subdir / "sample_in.txt"
    sample_output = problem_subdir / "sample_out.txt"
    problem_description = problem_subdir / "statement.txt"
    submit_folder = "./to_submit/"
    return Problem(
        problem_dir=problem_dir,
        problem_name=problem_name,
        problem_description=problem_description.read_text(),
        sample_input=sample_input.read_text(),
        sample_output=sample_output.read_text(),
        sample_input_path=Path(sample_input),
        sample_output_path=Path(sample_output),
        full_input_path=Path(problem_input),
        full_output_path=Path(problem_output),
        problem_input=problem_input,
        problem_output=problem_output,
        submit_folder=submit_folder,
    )


def load_problem(problem_name: str, problem_dir: pathlib.Path) -> Problem:
    problem_input = problem_dir / f"{problem_name}.in"
    problem_output = problem_dir / f"{problem_name}.out"
    sample_input = problem_dir / f"{problem_name}_sample_input.txt"
    sample_output = problem_dir / f"{problem_name}_sample_output.txt"
    problem_description = problem_dir / f"{problem_name}.md"
    submit_folder = "./to_submit/"
    return Problem(
        problem_dir=problem_dir,
        problem_name=problem_name,
        problem_description=problem_description.read_text(),
        sample_input=sample_input.read_text(),
        sample_output=sample_output.read_text(),
        problem_input=problem_input,
        problem_output=problem_output,
        submit_folder=submit_folder,
    )


def row_to_problem(row):
    return Problem(
        problem_dir=None,  
        problem_name=row['name'],
        problem_description=row['statement'],
        sample_input=row['sample_input'],
        sample_output=row['sample_output'],
        # full_input=row['input'],  
        # full_output=row['output'],
        problem_input=None,
        problem_output=None,
    )


def load_eval_dataset(ds, year="2023", round="practice"):
    def filter_function(example):
        return example['year'] == year and example['round'] == round # round1
    global filtered_ds
    filtered_ds = ds['full'].filter(filter_function)
    problems = [row_to_problem(row) for row in filtered_ds]  
    return problems

class Analysis(BaseModel):
    core_question: str = Field(..., description="Core question of the problem")
    problem_solving_info: List[str] = Field(
        ..., description="Problem-solving information related to the core question"
    )
    algorithm: str = Field(..., description="Algorithm to solve the problem")
    tutorial: str = Field(..., description="Tutorial on the algorithm")
    plan: str = Field(..., description="Step by step plan to solve the problem")
    pseudocode: str = Field(..., description="Pseudocode to solve the problem")

    @property
    def as_xml(self) -> str:
        return f"""
<core_question>
{self.core_question}
</core_question>
<problem_solving_info>
{self.problem_solving_info}
</problem_solving_info>
<algorithm>
{self.algorithm}
</algorithm>
<tutorial>
{self.tutorial}
</tutorial>
<plan>
{self.plan}
</plan>
<pseudocode>
{self.pseudocode}
</pseudocode>
"""


class Solution(Analysis):
    source_code: str = Field(
        ..., description="Valid Python3 sourcecode to solve the problem."
    )

    @property
    def as_xml(self) -> str:
        return f"""
<root>
{super().as_xml}
<source_code>
{self.source_code}
</source_code>
</root>
"""


class Reflection(BaseModel):
    reflection: str = Field(
        ...,
        description="Reflection on the problem, your solution, and the correct answer.",
    )
    keywords: List[str] = Field(
        ...,
        description="Keywords that describe the type of your errors from most general to most specific.",
    )
    step_by_step_solution: str = Field(
        ...,
        description="Step by step solution to the problem based on your knowledge of the correct answer.",
    )
    instructions: List[str] = Field(
        ...,
        description="Detailed instructions to help you correctly solve this problem in the future.",
    )
    general_advice: List[str] = Field(
        ...,
        description="General advice to help you solve similar types of problems in the future.",
    )

    @property
    def as_xml(self) -> str:
        return f"""
<root>
<reflection>
{self.reflection}
</reflection>
<keywords>
{self.keywords}
</keywords>
<step_by_step_solution>
{self.step_by_step_solution}
</step_by_step_solution>
<instructions>
{self.instructions}
</instructions>
<general_advice>
{self.general_advice}
</general_advice>
</root>
"""


def format_example(example: dict) -> str:
    formatted_doc = f"""
<problem>
<problem_statement>
{example['description']}
</problem_statement>
</problem>
<solution>
{example['code']}
</solution>
"""
    return formatted_doc


def format_examples(examples: List[dict], analyses: List[Analysis]) -> str:
    def format_question(example: dict) -> str:
        return f"""
<problem>
<problem_statement>
{example['description']}
</problem_statement>
</problem>
"""

    def format_solution(analysis: Analysis, example: dict) -> str:
        return f"""
<root>
{analysis.as_xml}
<source_code>
{example['code']}
</source_code>
</root>
"""

    messages = ""
    for example, analysis in zip(examples, analyses):
        messages += f"\n<example>{format_question(example)}\n{format_solution(analysis, example)}</example>\n"
    return messages.strip()
