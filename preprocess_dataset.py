from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.file import FlatReader
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = MockLLM()
Settings.text_splitter = SentenceSplitter()
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

import json 
import re 
#Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2", max_length=512)
PERSIST_DIR = './handbook'  # You can change this to any directory you prefer

def extract_index_from_tex(latex_text):
    # The regex pattern
    pattern = r'\\(key|chapter|section|subsection|subsubsection|index)\{([^}]*)\}'
    # Find all matches
    matches = re.findall(pattern, latex_text)
    # Display the results
    output = set()
    for command, content in matches:
        #print(f'Command: {command}, Content: {content}')
        output.add(content.lower().replace(' ','_'))

    pattern = r"e_maxx_link:\s*(.*)\n"
    # Find the match
    match = re.search(pattern, latex_text)

    # Output the match
    if match:
        output.add(match.group(1))
    
    return list(output)

def load_cpbook(directory):
    # Load the JSON file
    #directory = "data/cpbook_v2.json" #"data/cp_handbook.json"
    with open(directory, 'r') as file:
        data = json.load(file)  # Assuming 'data' is a list of entries
    # Process data into Documents
    documents = []
    for idx, entry in enumerate(data):
        # Adjust the keys based on your JSON structure
        content = entry.get('section_content') or entry.get('article') or entry.get('text') or ''
        title = entry.get('title') or entry.get('section_name') or ''
        tags = list(extract_index_from_tex(content))
        if not content:
            continue  # Skip entries without content
        # Create a Document
        doc_id = f"doc_{idx}"
        metadata = {
            'doc_id': doc_id,
            'title': title.lower().replace(' ','_'),
            'tags': tags,
        }
        document = Document(text=content, doc_id=doc_id, metadata=metadata)
        documents.append(document)
    return documents

def preprocess_dataset(): #only run once when buiding up the vector database
    reader = SimpleDirectoryReader(input_dir="data/")
    documents = reader.load_data()
    node_parser = SentenceSplitter()

    nodes = node_parser.get_nodes_from_documents(
        documents, show_progress=True
    )
    # Build the index
    index = VectorStoreIndex(nodes)
    # Save the index to the specified persist directory
    index.storage_context.persist(persist_dir=PERSIST_DIR)

a= load_cpbook("data/cp_handbook.json")
b= load_cpbook("data/cpbook_v2.json")
#preprocess_dataset()
print(f"vector database built in {PERSIST_DIR}")