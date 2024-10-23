"""from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.1", request_timeout=60.0)

response = llm.complete("What is the capital of France?")
print(response)
"""

import urllib.request
import json
from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training, load_problem_v2024
from pathlib import Path
from lib.utils import verify_code_syntax2, maybe_remove_backticks, extract_text, extract_python_code

from solution import Solution, SolutionManager
import random

problem_directory = "contestData"
problem_names = list_problem_names(problem_directory, "2024")
problem_list = []
for problem_name in problem_names:
    problem_list.append(load_problem_v2024(problem_name, Path(problem_directory)))

problem = problem_list[1]
print(problem.problem_name)

def query_model(messages, model="llama3.1", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": messages,
        "options": {
            "seed": seed,
            "temperature": 1,
            "num_ctx": 2048 # must be set, otherwise slightly random output
        }
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def self_refine():

    return out


def worker():

    return code

while True:
    
    prompt = """Solve the problem with lowest time complexity: {problem_description}"""
    seed = random.randint(1, 1000)
    messages = []
    messages.append(
            {
            "role": "user",
            "content": prompt.format(problem_description = problem.problem_description)
        }
    )

    result = query_model(messages)
    print(result)

    messages.append(
            {
            "role": "assistant",
            "content": result
        }
    )

    messages.append(
            {
            "role": "user",
            "content": "Write the python3 code based on the solution provided and please follow the input/output format in the problem statement. <source_code>[python3 code here, no external libary, use main function, and no examples or comments or explanations.]"
        }
    )

    result = query_model(messages)
    print(result)

    code = extract_text(result, '<source_code>')
    code = extract_python_code(code)
    code = maybe_remove_backticks(code)

    code_exec_flag, code = verify_code_syntax2(code)
    model_capability = "gpt3"

    print(code_exec_flag)

    if code_exec_flag:
        print("code executing...")    
        print('--code begin--\n', code, '\n--code end--')
        sol = Solution(code, problem.problem_name, problem.sample_input_path, problem.sample_output_path, problem.full_input_path, model_capability)
        sm = SolutionManager()
        a, b = sol.eval()
        print("sample report:", a.content)
        print("full report:",b.content)

        print("seed number:",seed)

        sm.add_solution(sol)

        if a.status == "passed" and b.status == "complete":
            print("success seed number:",seed)
            break