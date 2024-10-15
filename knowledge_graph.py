from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.readers.file import FlatReader
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
 
Settings.text_splitter = SentenceSplitter()
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
#Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2", max_length=512)
PERSIST_DIR = './handbook'  # You can change this to any directory you prefer

reader = SimpleDirectoryReader(input_dir="data/")
documents = reader.load_data()

from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

kg_extractor = SimpleLLMPathExtractor(
    llm=llm,
    max_paths_per_chunk=10,
    num_workers=4,
    show_progress=False,
)
 
print(f"vector database built in {PERSIST_DIR}")