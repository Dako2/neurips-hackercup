#the code evaluate the test.py 

from lib.utils import (
    create_logger,
    load_problem_from_folder,
    verify_code_syntax,
    extract_text,
    maybe_remove_backticks,
    save_to_disk,
)
from test import Trainer
from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training
from pathlib import Path

problem_directory = "/mnt/d/AIHackercup/dataset/2023/round2"
problem_names = list_problem_names(problem_directory, "2023")

logger = create_logger(f'logs/eval.log', 'eval')
 
for problem_name in problem_names:
    problem = load_problem_training(problem_name, Path(problem_directory))
    code = problem.best_code
    solution_guidelines = problem.solution
    
    problem_statement = problem.problem_description
    code = problem.best_code
    solution_guidelines = problem.solution
 
    logger.info(f"Solving {problem_name}")
 
    model_name = 'gpt4' #ranking powerful to less ['o1', 'gpt4', 'claude', 'gemini', 'gpt3.5'] from most capable to least capable 
    trainer1 = Trainer(model_name, problem,)

    sols = trainer1.battle_ground()
    trainer1.sm.to_submit('to_submit/')
    print(trainer1.sm.solution_manager)

