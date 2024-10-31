from transformers import GPT2Tokenizer

# Load pre-trained tokenizer (GPT-2 model)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

directory = "/mnt/d/AIHackercup/USACO/data/corpuses/cpbook_v2.json"
directory = "/mnt/d/AIHackercup/USACO/data/corpuses/cp_handbook.json"

# Load JSON data from a file
with open(directory, 'r') as file:
    text = file.read()
 
# Encode the text to tokens
tokens = tokenizer.encode(text)

# Number of tokens
num_tokens = len(tokens)

print(directory)
print(num_tokens)
