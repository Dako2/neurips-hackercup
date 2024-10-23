#streamlit run app.py

# 
import sys
import os
import pathlib
import logging
import time
import multiprocessing
from mtcs_v2 import MCTS_v2
from test import MCTS

# Third-party imports
import pyzipper  # Used if unzip functionality is required

# Custom module imports
from test import Trainer, output_format_indicator
from lib.utils import load_problem_from_folder, create_logger, unzip_questions

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from solution import Solution, SolutionManager

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Ensure necessary directories exist
os.makedirs("./to_submit", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Constants
CODE_PARENT_FOLDER = pathlib.Path('./Round1')  # TODO: Update as needed
ZIP_PATH = pathlib.Path('contestData.zip')#/mnt/c/user/tangqi/Downloads/
PASSWORD = "w3ak_password_for_practice"

def unzip_questions_if_needed():
    """
    Unzips the questions if they haven't been unzipped yet.
    """
    if not CODE_PARENT_FOLDER.exists() or not any(CODE_PARENT_FOLDER.iterdir()):
        logging.info("Unzipping questions...")
        try:
            unzip_questions(ZIP_PATH, PASSWORD, CODE_PARENT_FOLDER)
            logging.info("Questions unzipped successfully.")
        except ValueError as e:
            logging.error(f"Failed to unzip questions: {e}")
            sys.exit(1)
    else:
        logging.info("Questions already unzipped.")

def run_all_questions(problem_list):
    """
    Runs the main function for all selected questions using multiprocessing.
    """
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        executor.map(solver, problem_list)

    end_time = time.time()
    total_seconds = end_time - start_time
    logging.info(
        f"Total time taken for {len(problem_list)} questions: "
        f"{total_seconds:.0f} seconds / {total_seconds / 60:.1f} minutes"
    )

def solver(problem):
    logger = create_logger(f'logs/trainer_{problem.problem_name}.log', f'trainer{problem.problem_name}')
    logger.info(f"Solving {problem.problem_name}")
        
    _ = output_format_indicator(problem, logger)
        
    model_name = 'ollama' #ranking powerful to less ['o1', 'gpt4', 'claude', 'gemini', 'gpt3.5'] from most capable to least capable 
    trainer1 = Trainer(model_name, problem,)
    logger.info(trainer1.sm.solution_manager)
    sols = trainer1.battle_ground()
    trainer1.sm.to_submit('to_submit/')

def solver2(problem):
    logger = create_logger(f'logs/trainer_{problem.problem_name}.log', f'trainer{problem.problem_name}')
    logger.info(f"Solving {problem.problem_name}")
    
    model_name = 'gpt4' #ranking powerful to less ['o1', 'gpt4', 'claude', 'gemini', 'gpt3.5'] from most capable to least capable 
    #trainer1 = Trainer(model_name, problem,)
    #sols = trainer1.battle_ground()
    mcts = MCTS(model_name, problem)
    solution_node = mcts.mcts_trial(problem, max_steps=10)
    #print(mcts.sm.solution_manager)
    logger.info(mcts.sm.solution_manager)
    mcts.sm.to_submit('to_submit/')
 

if __name__ == "__main__":
    #unzip_questions_if_needed()
    
    from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training, load_problem_v2024
    from pathlib import Path

    """
    problem_directory = "dataset/2023/round2"
    problem_names = list_problem_names(problem_directory, "2023")
    problem_list = []
    for problem_name in problem_names:
        problem_list.append(load_problem_training(problem_name, Path(problem_directory)))
    """

    problem_directory = "contestData"
    problem_names = list_problem_names(problem_directory, "2024")
    problem_list = []
    for problem_name in problem_names:
        problem_list.append(load_problem_v2024(problem_name, Path(problem_directory)))

    # Uncomment the following line if unzipping is needed
    if len(sys.argv) > 1:
        # Convert command-line arguments to zero-based indices
        selected_questions = [int(arg) - 1 for arg in sys.argv[1:] if arg.isdigit()]
        if selected_questions:
            problem_list = [problem_list[x] for x in selected_questions]
            run_all_questions(problem_list)
        else:
            logging.warning("No valid problems selected to solve.")
    else:
        logging.warning("No problem numbers provided. Usage: python script.py <problem_numbers>")



