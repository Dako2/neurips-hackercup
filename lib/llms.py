import os
from anthropic import Anthropic
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import ollama
import logging 
import random 
from typing import List, Optional, Dict, Any
import asyncio
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API key not found in environment variables.")
openai.api_key = OPENAI_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Anthropic API key not found in environment variables.")

async def fetch_gemini_response(messages, temperature, max_tokens):
    """Call the Gemini model."""
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")

    response = gemini_model.generate_content(transform_to_gemini(messages),)
    if response:
        response = response.text
    else:
        response = "No response received from Gemini."
    return response

async def fetch_o1_response(messages, max_tokens):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
                model="o1-mini",
                messages=messages,
            )
    return response.choices[0].message.content

async def fetch_openai_response(messages, temperature, max_tokens):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=temperature,
                tool_choice=None
            )
    return response.choices[0].message.content

async def fetch_gpt3_response(messages, temperature, max_tokens):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                tool_choice=None
            )
    return response.choices[0].message.content

async def fetch_anthropic_response(messages, temperature, max_tokens):
    anthro_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    """Call the Anthropic (Claude) model."""
    if not anthro_client:
        raise RuntimeError("Anthropic client is not initialized.")
    
    message = anthro_client.messages.create(
        model="claude-3-opus-20240229",
        temperature=temperature,  # Added temperature parameter
        messages=messages,
        max_tokens=4096,
    )
    response = message.content
    return response[0].text

async def generate_response(model: str, messages, temperature: Optional[float], max_tokens: int) -> dict:
    if model.lower() in ['gpt4']:
        return await fetch_openai_response(messages, temperature, max_tokens)
    elif model.lower() in ['gpt3.5']:
        return await fetch_gpt3_response(messages, temperature, max_tokens)
    elif model.lower() in ['anthropic', 'claude']:
        return await fetch_anthropic_response(messages, temperature, max_tokens)
    elif model.lower() in ['o1']:
        return await fetch_o1_response(messages, max_tokens)
    elif model.lower() in ['gemini']:
        return await fetch_gemini_response(messages, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model: {model}")

async def mcts_openai_messages_v2_async(messages, temperature=1, model_list = ['o1','gpt4'], n = 2):
    #if n = 1, select the first one out of model_list and generate answer
    #if n = 2, random select 2 out of model_list and generate answers
    #if n = 5, random select 5 out of model_list and generate answers
    #asyncio return the two responses in the list
    selected_models = random.sample(model_list, n)

    # Define max_tokens or make it a parameter
    max_tokens = 1024

    # Create coroutines for each selected model
    tasks = [
        generate_response(model, messages, temperature, max_tokens)
        for model in selected_models
    ]
        # Execute all tasks concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions and collect successful responses
    final_responses = []
    for model, resp in zip(selected_models, responses):
        if isinstance(resp, Exception):
            # Handle or log the exception as needed
            print(f"Error with model {model}: {resp}")
            final_responses.append({'model': model, 'error': str(resp)})
        else:
            final_responses.append({'model': model, 'response': resp})

    return final_responses

def mcts_openai_messages_v2(
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    model_list: List[str] = ['openai', 'gpt4', 'anthropic'],
    n: int = 2
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for the asynchronous mcts_openai_messages_v2_async function.

    :param messages: The conversation or prompts to send to the models.
    :param temperature: Controls the randomness of the model's output.
    :param model_list: List of available models to choose from.
    :param n: Number of models to select and generate responses from.
    :return: List of responses from the selected models.
    """
    return asyncio.run(mcts_openai_messages_v2_async(messages, temperature, model_list, n))


def transform_to_gemini(messages_chatgpt):
    messages_gemini = []
    system_promt = ''
    for message in messages_chatgpt:
        if message['role'] == 'system':
            system_promt = message['content']
        elif message['role'] == 'user':
            messages_gemini.append({'role': 'user', 'parts': [message['content']]})
        elif message['role'] == 'assistant':
            messages_gemini.append({'role': 'model', 'parts': [message['content']]})
    if system_promt:
        messages_gemini[0]['parts'].insert(0, f"*{system_promt}*")
    return messages_gemini
    
class LLM:
    def __init__(self, model_name="anthropic", logger=None) -> None:
        self.prompt = ""
        self.response = ""
        self.model_name = model_name.lower()  # Normalize model name to lower case

        # Initialize the respective model client based on model_name
        self.anthro_client = None
        self.gemini_model = None
        self.openai_client_initialized = False
        self.gemini_model_initialized = False
        self.anthropic_model_initialized = False
        if not logger:
            self.logger = logging
        else:
            self.logger = logger
        self.client = None
        if self.model_name == "anthropic":
            self.initialize_anthropic()
        elif self.model_name == "gpt4" or "o1" or "o1-mini" or "gpt3.5":
            self.initialize_openai()
        elif self.model_name == "gemini":
            self.initialize_gemini()
        elif self.model_name == "codegemma" or "llama3":
            self.initialize_ollama(self.model_name)
        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

    def initialize_ollama(self, model_name):
        # Prepare the messages for the model
        messages = [
            {
                'role': 'user',
                'content': ''
            }
        ]
        ollama.chat(model=model_name, messages=messages)
  
    def ollama(self, prompt):
        # Prepare the messages for the model
        messages = [
            {
                'role': 'user',
                'content': prompt
            }
        ]
        self.response = ollama.chat(model=self.model_name, messages=messages)
        return self.response['message']['content']
    
    def ollama_messages(self, messages):
        # Prepare the messages for the model
        self.response = ollama.chat(model=self.model_name, messages=messages)
        return self.response['message']['content']

    def initialize_anthropic(self):
        """Initialize the Anthropic (Claude) client."""
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        if not ANTHROPIC_API_KEY:
            raise RuntimeError("Anthropic API key not found in environment variables.")
        self.anthro_client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def initialize_openai(self):
        """Initialize the OpenAI client."""
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key not found in environment variables.")
        openai.api_key = OPENAI_API_KEY
        self.client = openai.OpenAI()
        self.openai_client_initialized = True
        
    def openai(self, prompt, temperature=0.7):
        """Call the OpenAI (GPT-4) model using the new ChatCompletion API."""
        if not self.openai_client_initialized:
            raise RuntimeError("OpenAI client is not initialized.")
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,  # Added temperature parameter
            #max_tokens=1024
        )        
        self.response = response.choices[0].message.content.strip()
        self.logger.info(f"\n\openai reponse:{self.response}")
        return self.response

    def openai_ft_messages(self, messages, temperature=1):
        """Call the OpenAI (GPT-4) model using the new ChatCompletion API."""
        if not self.openai_client_initialized:
            raise RuntimeError("OpenAI client is not initialized.")
        
        response = self.client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:aihackercup:ADjyWVOK",#"gpt-4o-2024-08-06",
            messages=messages,
            temperature=temperature,  # Added temperature parameter
            #max_tokens=1024
        )        
        # Extract the response content
        self.response = response.choices[0].message.content.strip()
        self.logger.info(f"\n\openai reponse:{self.response}")
        return self.response
    
    def openai_messages(self, messages, temperature=None, model_name="gpt-4o-2024-08-06"):
        """Call the OpenAI (GPT-4) model using the new ChatCompletion API."""
        if not self.openai_client_initialized:
            raise RuntimeError("OpenAI client is not initialized.")
        
        if temperature:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,  # Added temperature parameter
                #max_tokens=1024
            )        
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                # Added temperature parameter
                #max_tokens=1024
            )        
        # Extract the response content
        self.response = response.choices[0].message.content.strip()
        #self.logger.info(f"\n********Openai reponse:{self.response}\n*********")
        return self.response
    
    def mcts_openai_messages(self, messages, temperature=None, model_name="gpt-4o-2024-08-06", n = 1):
        """Call the OpenAI (GPT-4) model using the new ChatCompletion API."""
        if not self.openai_client_initialized:
            raise RuntimeError("OpenAI client is not initialized.")
        
        if temperature:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,  # Added temperature parameter
                max_tokens=1024,
                n = n,
            )        
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                # Added temperature parameter
                max_tokens=1024,
                n = n,
            )        
        # Extract the response content
        self.response = response
        #self.logger.info(f"\n********Openai reponse:{self.response}\n*********")
        return self.response
    
    def initialize_gemini(self):
        """Initialize the Gemini client."""
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise RuntimeError("Gemini API key not found in environment variables.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        self.gemini_model_initialized = True
        
    def anthropic(self, prompt, temperature=1):
        """Call the Anthropic (Claude) model."""
        if not self.anthro_client:
            raise RuntimeError("Anthropic client is not initialized.")
        
        message = self.anthro_client.messages.create(
            #max_tokens=1024,
            model="claude-3-opus-20240229",
            temperature=temperature,  # Added temperature parameter
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            
        )
        self.response = message['completion']
        return self.response

    def anthropic_messages(self, messages, temperature=1):
        """Call the Anthropic (Claude) model."""
        if not self.anthro_client:
            raise RuntimeError("Anthropic client is not initialized.")
        
        message = self.anthro_client.messages.create(
            model="claude-3-opus-20240229",
            temperature=temperature,  # Added temperature parameter
            messages=messages,
            max_tokens=4096,
        )
        print(message)
        self.response = message.content
        self.logger.info(f"\n\gemini reponse:{self.response}")
        return self.response[0].text

    def gemini(self, prompt, temperature=0.7):
        """Call the Gemini model."""
        if not self.gemini_model:
            raise RuntimeError("Gemini model is not initialized.")
        
        response = self.gemini_model.generate_content(prompt,)
        if response:
            self.response = response.text
        else:
            self.response = "No response received from Gemini."
        self.logger.info(f"\n\gemini reponse:{self.response}")
        return self.response

    def run(self, prompt, temperature=0.7):
        """Run the selected model."""
        if self.model_name == "anthropic":
            return self.anthropic(prompt, temperature)
        elif self.model_name == "gemini":
            return self.gemini(prompt, temperature)
        elif self.model_name == "openai":
            return self.openai(prompt, temperature)
        elif self.model_name == "codegemma" or "llama3.1":
            return self.ollama(prompt)
        else:
            return "Model not supported. Please choose from 'anthropic', 'gemini', or 'openai'."
    
    def run_messages(self, messages, temperature=0.7):
        if self.model_name == "gpt4":
            return self.openai_messages(messages, temperature, "gpt-4-turbo-preview")#gpt-4-turbo-preview#gpt-4o-2024-08-06
        elif self.model_name == "gpt3.5":
            return self.openai_messages(messages, None, "gpt-3.5-turbo")
        elif self.model_name == "o1-mini":
            return self.openai_messages(messages, None, "o1-mini-2024-09-12")
        elif self.model_name == "o1":
            return self.openai_messages(messages, None, "o1-mini")
        elif self.model_name == "gemini":
            return self.gemini(transform_to_gemini(messages), temperature)
        elif self.model_name == "anthropic":
            return self.anthropic_messages(messages, temperature)
        elif self.model_name == "llama3.1" or "codegemma":
            return self.ollama_messages(messages)
        else:
            print(self.model_name)
            raise ValueError("model selection error in run_messages")
        
# Example usage
if __name__ == "__main__":
    #- Key constraints and unique conditions
    
    summarizer_prompt="""Please provide a concise summary (within 60 words) that captures:
    **Output**: Use the following xml output format - 
    - The provided solution types (e.g., graph traversal, dynamic programming); 
    - The main challenges of solving the problem;
    - The key coding/implementation ideas;
    - Provide rationale for the solution;
    - A list of tags or keywords for search;

    ##Problem Statement:
    {problem_statement}

    ##Code_solution:
    {code}

    ##Solution Guidelines:
    {solution_guidelines}
    
    **Output**
    """

    #response = llm.run(prompt)    
    messages=[
        {
            "role": "user",
            "content": "hi!!",
        }
    ]
    
    res = mcts_openai_messages_v2(messages, temperature=1, model_list = ['gemini','gpt4','anthropic'], n = 2)

    """
    from utils import load_problem_from_folder, list_problem_names, load_problem_training
    from pathlib import Path
    
    problem_directory = "/mnt/d/AIHackercup/dataset/2023/round2"
    problem_names = list_problem_names(problem_directory, "2023")

    for problem_name in problem_names[:1]:
        problem = load_problem_training(problem_name, Path(problem_directory))
        code = problem.best_code
        solution_guidelines = problem.solution
        
        problem_statement = problem.problem_description
        code = problem.best_code
        solution_guidelines = problem.solution
        
        for model in ["gpt4",]:
            llm = LLM(model_name=model)
            #response = llm.run(prompt)    
            messages=[
                {
                    "role": "user",
                    "content": summarizer_prompt.format(
                        problem_statement = problem_statement,
                        code = code,
                        solution_guidelines = solution_guidelines,
                    ),
                }
            ]
            response = llm.mcts_openai_messages_v2(messages, n=1)
            print(f"##{model}:",response.choices[0].message.content.strip())
            print(f"##{model}:",response.choices[1].message.content.strip())
            #print(type(transform_to_gemini(messages)))
    """