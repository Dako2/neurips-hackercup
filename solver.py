
from lib.utils import load_problem_from_folder, list_problem_names, load_problem_training, load_problem_v2024
from pathlib import Path
from lib.utils import (
    create_logger,
    load_problem_from_folder,
    verify_code_syntax,
    extract_text,
    maybe_remove_backticks,
    save_to_disk,)
from lib.llms import LLM
from solution import Solution, SolutionManager
from config import SELECT_LANGUAGE, EXTRACT_CODE_PROMPT, ANALYZER_PROMPT
simple_initial_advisor_prompt = """Solve codeforce problem. 
- Ensure that the code includes a main function and is structured so that the program is executable when run, following the conventions of {selected_language}.
- Do not use any external libraries; only stick to {selected_language} standard library
Hint: Highly optimized and suited for large inputs, using advanced techniques like Fenwick Trees and binary search.
- Do NOT USE nested loops / iterations
- DO NOT USE inefficient pair counting
- DO NOT USE Brute force
Problem: {problem}"""

class Solver():
    def __init__(self, logger = None):
        self.logger = logger
        if not self.logger:
            self.logger=create_logger("temp.log","temp")
        self.memory = []
        self.sm = SolutionManager()        
        self.strong_llm = LLM(model_name="gpt4")
        self.fast_llm = LLM(model_name="gpt4")

    def analyzer(self,problem):
        prompt = ANALYZER_PROMPT.format(problem_statement=problem.problem_description)
        messages = [
            {
                'role': 'user',
                'content': prompt + Path("lib/analyzer_prompt.txt").read_text()
            }
        ]
        self.memory += [
            {
                'role': 'user',
                'content': ANALYZER_PROMPT.format(problem_statement="")
            }
        ]
        output = self.strong_llm.run_messages(messages=messages)
        self.memory += [
            {
                'role': 'assistant',
                'content': output
            }
        ]
        return output
        
    def _coder(self, out0, selected_language):
        out0 = extract_text(out0, '<source_code>')
        out0 = maybe_remove_backticks(out0)
        extract_code_message = EXTRACT_CODE_PROMPT.format(
            output=out0,
            selected_language=selected_language)
        messages = [
            {
                'role': 'user',
                'content': extract_code_message
            }
        ]
        code = self.fast_llm.run_messages(messages=messages)
        if '<source_code>' in code:
            code = extract_text(code, '<source_code>')

        code = maybe_remove_backticks(code)
        return code
    
    def _reflector(self,feedback):
        self.memory += [{'role': 'user', 'content': feedback}]
        out = self.strong_llm.run_messages(messages=self.memory)
        code = self._coder(out, SELECT_LANGUAGE)
        self.memory += [{'role': 'assistant', 'content':code}]
        return code

    def zero_shot(self, problem): #implement the code; fixed context length simple response
        """Processes assistant output to extract and verify the source code."""
        prompt = simple_initial_advisor_prompt.format(problem=problem.problem_description, selected_language="cpp")
        self.memory += [{'role': 'user', 'content': prompt}]
        out = self.strong_llm.run_messages(messages=self.memory)
        code = self._coder(out, SELECT_LANGUAGE)
        self.memory += [{'role': 'assistant', 'content':code}]
        s = Solution(code, problem, self.strong_llm.model_name, "zeroshot", lang=SELECT_LANGUAGE) #py, cpp
        self.sm.add_solution(s)
        s.eval(logger=self.logger)
        return code
    
    def self_reflection(self,problem,n=3):
        for i in range(n):
            print(self.memory)
            if i == 0:
                code = self.zero_shot(problem)
                s = Solution(code, problem, self.strong_llm.model_name, "zeroshot", lang=SELECT_LANGUAGE) #py, cpp
                testreport, fullreport = s.eval(logger=self.logger)
                self.sm.add_solution(s)
                if s.to_submit_or_not:
                    break
                continue
            code = self._reflector(f"{problem.problem_name}:\n - Sample test: {testreport.message}\n - Full test: {fullreport.as_xml}")   
            s = Solution(code, problem, self.strong_llm.model_name, "zeroshot", lang=SELECT_LANGUAGE) #py, cpp
            testreport, fullreport = s.eval(logger=self.logger)
            self.sm.add_solution(s)
            if s.to_submit_or_not:
                break
        return code
    
if __name__ == '__main__':

    problem_directory = "dataset/2024/Round2"
    problem_names = list_problem_names(problem_directory, "2024")
    problem_list = []
    for problem_name in problem_names:
        problem_list.append(load_problem_v2024(problem_name, Path(problem_directory)))
    problem = problem_list[1]
    print(problem.problem_name)

    solver = Solver(problem)
    code = solver.zero_shot(problem)
    code1 = solver.self_reflection(problem,n=2)
    #code = Path("generated//Bunny_Hopscotch_1730085454_score.cpp").read_text()
    #code = Path("/Users/dako22/Downloads/bunny_hopscotch__molamola_source_code.txt").read_text() + '\n'
    solver.sm.to_submit("to_submit/")


    