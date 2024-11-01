import os
from anthropic import Anthropic
import openai
import google.generativeai as genai
from dotenv import load_dotenv

import logging 
import random 
from typing import List, Optional, Dict, Any
import asyncio
from openai import AsyncOpenAI

from llama_index.llms.ollama import Ollama
import json
import urllib
import urllib.request

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

anthro_client = Anthropic(api_key=ANTHROPIC_API_KEY)
client = AsyncOpenAI()

async def fetch_gemini_response_n(messages, n):
    """Call the Gemini model."""
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro",generation_config=genai.types.GenerationConfig(candidate_count=n))

    response = gemini_model.generate_content(transform_to_gemini(messages),)
    return response

async def fetch_gemini_response(messages):
    """Call the Gemini model."""
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    response = gemini_model.generate_content(transform_to_gemini(messages),)
    if response:
        response = response.text
    else:
        response = "No response received from Gemini."
    return response

async def fetch_o1_response(model_name, messages):
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
    return [response.choices[0].message.content]

async def fetch_openai_response_n(model_name, messages, temperature, n, seed):
    if seed:
        response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    tool_choice=None,
                    n=n,
                    seed=seed,
                    #response_format={ "type": "json_object" },
                )
    else:
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            tool_choice=None,
            n=n,
            #response_format={ "type": "json_object" },
        )
    output = [response.choices[i].message.content for i in range(n)]
    return output

async def fetch_anthropic_response(messages, temperature):
    """Call the Anthropic (Claude) model."""
    if not anthro_client:
        raise RuntimeError("Anthropic client is not initialized.")
    
    message = anthro_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        temperature=temperature,  # Added temperature parameter
        messages=messages,
        max_tokens=4096,
    )
    response = message.content
    return response[0].text

async def generate_response_n(model: str, messages, temperature: Optional[float], n: int, seed=None) -> dict:
    if model.lower() in ['gpt4']:
        return await fetch_openai_response_n("gpt-4", messages, temperature, n, seed)
    elif model.lower() in ['gpt3.5']:
        return await fetch_openai_response_n("gpt-3.5-turbo", messages, temperature, n, seed)
    elif model.lower() in ['anthropic', 'claude']:
        return await fetch_anthropic_response(messages, temperature)
    elif model.lower() in ['o1']:
        return await fetch_o1_response('o1-mini', messages)
    elif model.lower() in ['gemini']:
        return await fetch_gemini_response_n(messages, n)

async def mcts_openai_messages_v2_async(messages, temperature, model_list, n):
    # Create coroutines for each selected model
    tasks = [
        generate_response_n(model, messages, temperature, n)
        for model in model_list
    ]
        # Execute all tasks concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions and collect successful responses
    final_responses = []
    for model, resp in zip(model_list, responses):
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
    model_list: List[str] = ['gpt4', 'anthropic','gemini'],
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
        self.ollama_llm = None
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
        elif self.model_name == "ollama" or "llama3.1":
            self.ollama_llm = self.initialize_ollama(self.model_name)

        else:
            raise ValueError(f"Model '{self.model_name}' is not supported.")


    def initialize_ollama(self, model_name="llama3.1"):
        # Prepare the messages for the model
        llm = Ollama(model="llama3.1", request_timeout=60.0)
        response = llm.complete("be a code helper.")
        #print(response)
        return llm
  
    def ollama(self, prompt):
        if not self.ollama_llm:
            self.ollama_llm = self.initialize_ollama()
        # Prepare the messages for the model
        response = self.ollama_llm.complete(prompt)
        return response
    
    def ollama_messages(self, messages):
        # Prepare the messages for the model
        prompt = json.dumps(messages)
        # Prepare the messages for the model
        return self.ollama(prompt)

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
        return self.response
    
    def openai_messages_seed(self, messages, seed=200, model_name="gpt-4o-2024-08-06"):
        """Call the OpenAI (GPT-4) model using the new ChatCompletion API."""
        if not self.openai_client_initialized:
            raise RuntimeError("OpenAI client is not initialized.")
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            seed=seed,
            temperature=1,  # Added temperature parameter
            max_tokens=2048,
        )
        self.response = response.choices[0].message.content.strip()
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
            return self.openai_messages(messages, temperature, "gpt-4o-2024-08-06")#gpt-4-turbo-preview#gpt-4o-2024-08-06
        elif self.model_name == "gpt3.5":
            return self.openai_messages(messages, temperature, "gpt-3.5-turbo")
        elif self.model_name == "o1-mini":
            return self.openai_messages(messages, None, "o1-mini-2024-09-12")
        elif self.model_name == "o1":
            return self.openai_messages(messages, None, "o1-mini")
        elif self.model_name == "gemini":
            return self.gemini(transform_to_gemini(messages), temperature)
        elif self.model_name == "anthropic":
            return self.anthropic_messages(messages, temperature)
        elif self.model_name == "llama3.1" or "ollama":
            return self.ollama_messages(messages)
        else:
            print(self.model_name)
            raise ValueError("model selection error in run_messages")

# Function to query the LLM (your 'query_model' function)
def call_llama(prompt, model="llama3.1", seed=None, url="http://localhost:11434/api/chat"):
    """
    Calls the LLM with the given prompt and returns the response.
    """
    if seed is None:
        seed = random.randint(1, 1000)

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    data = {
        "model": model,
        "messages": messages,
        "options": {
            "seed": seed,
            "temperature": 1,
            "num_ctx": 2048  # Ensure consistent output
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    try:
        with urllib.request.urlopen(request) as response:
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_data += response_json["message"]["content"]
    except Exception as e:
        raise ImportError
    #logger.info(f"LLM Response: {response_data}")
    return response_data.strip()
 
# Define the main function
import time 
async def main():
    # Define the parameters
    start_timer = time.time()
    messages = [{"role": "user", "content": "List 20 difficult algorithm names used in competitive coding in the json format: {'algorithms':[...]}"}]
    temperature = 1
    n = 2  # Number of completions you want to fetch

    # Call the fetch_openai_response_n function and await its result
    model_name = 'o1'
    seed = None #random.randint(0,1000)
    responses = await generate_response_n(model_name, messages, temperature, n, seed)

    print(f"time {model_name}: {time.time()-start_timer}")
    # Create a set to store unique algorithm names
    algorithm_names = set()
    # Write the set to a file
    with open("algorithm_names1.txt", "a") as file:
        for name in responses:
            file.write(f"{name}\n")

# Run the main function using asyncio
if __name__ == "__main__":
    asyncio.run(main())