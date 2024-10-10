import sys
import os
import pathlib
import logging
import time
import multiprocessing

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
ZIP_PATH = pathlib.Path('/Users/dako22/Downloads/contestData (1).zip')
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

def run_all_questions(selected_questions):
    """
    Runs the main function for all selected questions using multiprocessing.
    """
    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        executor.map(main, selected_questions)

    end_time = time.time()
    total_seconds = end_time - start_time
    logging.info(
        f"Total time taken for {len(selected_questions)} questions: "
        f"{total_seconds:.0f} seconds / {total_seconds / 60:.1f} minutes"
    )

def main(selected_question_index):
    problem_names = [f.name for f in CODE_PARENT_FOLDER.iterdir() if f.is_dir()]
    try:
        problem_name = problem_names[selected_question_index]
        logger.info(f"Processing problem: {problem_name}")
    except IndexError:
        logging.error(f"Selected question index {selected_question_index + 1} is out of range.")
        return
    logger = create_logger(f'logs/trainer_{selected_question_index}.log', f'trainer{selected_question_index}')
    problem = load_problem_from_folder('2024', CODE_PARENT_FOLDER, problem_name, logger)
    logger.info(f"Solving {problem_name}")
    logger.info(f"Plans {plans}")
    # List of model-capability pairs along with the method to be called
    plans = [
        ('gpt3.5', "solve_problem_pro"),  # 'openai' model with 'gpt4' capability
        #('gpt4', "reflection_pro"),  # 'gemini' model with 'chain_of_thoughts' method
        #('gpt4', "chain_of_thoughts"),  # 'gemini' model with 'chain_of_thoughts' method
    ]
    logger.info(f"Solving {problem_name}")
    solver(problem, plans, logger)

def solve_problem(trainer_method, problem, model_name, model_capability_ranking, sm, logger):
    logger.info(f"Starting to solve with {model_name}, using {trainer_method.__name__}...")
    try:
        code = trainer_method()
        if not code:
            logger.warning(f"No solutions returned by {trainer_method.__name__} for problem {problem.problem_name}")
        # Evaluate and add solutions
        s = Solution(code, problem.problem_name, problem.sample_input_path, problem.sample_output_path, problem.full_input_path, model_name)
        print("s", flush=True)
        testreport, full_testreport = s.eval()
        logger.info(f"Evaluating solution: {testreport}")
        sm.add_solution(s)
        logger.info(f"Solution added for {problem.problem_name}")
    except Exception as e:
        logger.error(f"Error solving {problem.problem_name} with {model_name}: {e}")

def solver(problem, plans, logger):
    _ = output_format_indicator(problem, logger)
    sm = SolutionManager()
    logger.info("solution manager created")
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the thread pool
        futures = []
        for model_name, method in plans:
            # Submit the bound method (solve_problem_pro or chain_of_thoughts)
            futures.append(executor.submit(solve_problem, model_name, method, sm))
        # Wait for all threads to finish
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # This will raise an exception if something went wrong in the thread
            except Exception as e:
                logger.error(f"Error in thread: {e}")
    # Once all threads are done, submit the solutions
    logger.info(f"{sm.solution_manager}")
    logger.info(f"{problem.problem_name} problem solved")
    sm.to_submit('to_submit/')

def solve_problem(model_name, method, sm):
    # Call the provided method directly (either solve_problem_pro or chain_of_thoughts)
        # Initialize the trainer
    trainer = Trainer(model_name, problem)
    solutions = trainer.run(method)
    # Evaluate the solution
    for s in list(solutions):
        testreport, full_testreport = s.eval()
        # Add the solution to the solution manager
        sm.add_solution(s)
        
if __name__ == "__main__":
    unzip_questions_if_needed()

    # Uncomment the following line if unzipping is needed
    if len(sys.argv) > 1:
        # Convert command-line arguments to zero-based indices
        selected_questions = [int(arg) - 1 for arg in sys.argv[1:] if arg.isdigit()]
        if selected_questions:
            run_all_questions(selected_questions)
        else:
            logging.warning("No valid problems selected to solve.")
    else:
        logging.warning("No problem numbers provided. Usage: python script.py <problem_numbers>")



