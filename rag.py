from llama_index.core import (
    StorageContext,
    Settings,
)

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.llm = MockLLM()
Settings.text_splitter = SentenceSplitter()
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
#Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2", max_length=512)
PERSIST_DIR = './handbook'  # You can change this to any directory you prefer

class RAG:
    def __init__(self):
        # Define a persist directory
        # Load the index from the same persist directory
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        self.index = load_index_from_storage(storage_context)
     
    #TODO
    def retrieve(self,question,similarity_top_k=3):#node level
        query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)  # Adjust 'top_k' as needed
        # Perform the query
        response = query_engine.query(question)
        output = ""
        for node_with_score in response.source_nodes:
            output += node_with_score.node.text
        return response

#rag = RAG()
#b = rag.retrieve("Sieve of Eratosthenes", similarity_top_k=3)


