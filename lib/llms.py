import os
from anthropic import Anthropic
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import ollama
import logging 

# Load environment variables from .env file
load_dotenv()

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
        elif self.model_name == "codegemma" or "llama3.1":
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
    
    def initialize_gemini(self):
        """Initialize the Gemini client."""
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise RuntimeError("Gemini API key not found in environment variables.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            
    def anthropic(self, prompt, temperature=0.7):
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

    def anthropic_messages(self, messages, temperature=0.7):
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
            return self.openai_messages(messages, temperature, "gpt-4o-2024-08-06")
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
        else:
            print(self.model_name)
            raise ValueError("model selection error in run_messages")
      
# Example usage
if __name__ == "__main__":
    
    llm = LLM(model_name="anthropic")
    prompt = "what day is it today?"
    #response = llm.run(prompt)    
    messages=[
        {
            "role": "user",
            "content": "hello",
        }
    ]
    response = llm.run_messages(messages)
    print(response)
    #print(type(transform_to_gemini(messages)))