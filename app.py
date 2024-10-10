import streamlit as st
from test import Trainer, output_format_indicator
from solution import SolutionManager, Solution
from lib.utils import create_logger, load_problem_from_folder
from pathlib import Path

# Set up the page layout
st.set_page_config(layout="wide")

# Initialize session state for generated code and evaluation reports
if 'generated_code' not in st.session_state:
    st.session_state['generated_code'] = ''
if 'test_report' not in st.session_state:
    st.session_state['test_report'] = ''
if 'evaluation_report' not in st.session_state:
    st.session_state['evaluation_report'] = ''

# Title of the application
st.title("AI Problem Solver")

# Define the directory containing the problems
problem_directory = 'Round1/'  # Adjust the path as needed

# Get the list of problem names from the directory
problem_paths = [p for p in Path(problem_directory).iterdir() if p.is_dir()]
problem_names = [p.name for p in problem_paths]

# Create a dropdown for selecting the problem
selected_problem_name = st.selectbox("Select a Problem:", problem_names)

if selected_problem_name:
    # Create a logger
    logger = create_logger(f'logs/trainer_{selected_problem_name}.log', 'trainer')

    # Load the problem using your original function
    problem = load_problem_from_folder('2024', problem_directory, selected_problem_name, logger)

    # Initialize SolutionManager
    sm = SolutionManager()

    # **Place Buttons at the Top**
    # We can use a container to hold the buttons and position them at the top
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        generate_button = st.button("Generate Solution")
    with button_col2:
        evaluate_button = st.button("Evaluate Code")

    # Left Column: Problem Input
    left_col, right_col = st.columns(2)

    with left_col:
        st.header("Problem Statement")
        # Display the problem description in a text area
        problem_description = st.text_area("Problem Statement:", problem.problem_description, height=300)

        st.header("Sample Input and Output")
        sample_input = st.text_area("Sample Input:", problem.sample_input, height=100)
        sample_output = st.text_area("Sample Output:", problem.sample_output, height=100)

    # Function to generate solution
    def generate_solution():
        if problem_description and sample_input and sample_output:
            # Update the problem instance with any changes made in the text areas
            problem.problem_description = problem_description
            problem.sample_input = sample_input
            problem.sample_output = sample_output

            # Save sample input and output to files
            problem.sample_input_path.write_text(sample_input)
            problem.sample_output_path.write_text(sample_output)

            # Determine if exact output format is required
            _ = output_format_indicator(problem, logger)

            # Initialize Trainer
            model_name = 'gpt4'  # Adjust as needed
            trainer = Trainer(model_name, problem)

            # Run the problem-solving method
            sols = trainer.solve_problem_pro()
            if sols:
                s = sols[0]  # Take the first solution
                # Update session state with generated code
                st.session_state['generated_code'] = s.code

                # Evaluate the solution
                testreport, full_testreport = s.eval()

                # Update session state with evaluation results
                st.session_state['test_report'] = testreport.content
                st.session_state['evaluation_report'] = (
                    full_testreport.content if full_testreport else "No full test report available."
                )

                # Optionally, handle submission of solutions
                sm.add_solution(s)
                sm.to_submit('to_submit/')
            else:
                st.error("No solution was generated.")
        else:
            st.error("Please provide the problem statement, sample input, and sample output.")

    # Function to evaluate code
    def evaluate_code():
        # Get the code from the code editor using its key
        edited_code = st.session_state.get('code_editor', '')
        if edited_code.strip() == '':
            st.error("The code editor is empty. Please generate or enter code to evaluate.")
        else:
            # Update the session state with the edited code
            st.session_state['generated_code'] = edited_code

            # Create a new Solution instance with the edited code
            s = Solution(
                code=edited_code,
                problem_name=problem.problem_name,
                sample_input_path=problem.sample_input_path,
                sample_output_path=problem.sample_output_path,
                full_input_path=problem.full_input_path,
                model_capability='User Edited'
            )
            testreport, full_testreport = s.eval()

            # Update session state with evaluation results
            st.session_state['test_report'] = testreport.content
            st.session_state['evaluation_report'] = (
                full_testreport.content if full_testreport else "No full test report available."
            )

            # Optionally, handle submission of solutions
            sm.add_solution(s)
            sm.to_submit('to_submit/')

    # Handle button clicks
    if generate_button:
        generate_solution()

    if evaluate_button:
        evaluate_code()

    # Right Column: Code Editor and Evaluation
    with right_col:
        st.header("Generated Code Solution")
        code_editor = st.text_area(
            "Code Solution:",
            value=st.session_state['generated_code'],
            height=300,
            key='code_editor'  # Assign a unique key
        )

        st.header("Solution Log")
        log_output = st.empty()  # Placeholder for log output

        st.header("Evaluation Report")
        evaluation_report = st.empty()  # Placeholder for evaluation report

    # Display logs and evaluation reports
    if st.session_state['test_report']:
        log_output.text(st.session_state['test_report'])

    if st.session_state['evaluation_report']:
        evaluation_report.text(st.session_state['evaluation_report'])
