#rag-experiment-2.py

#simple rag

"""#simplest parser
# # OPTION 2: Use SimpleDirectoryReader
from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings


documents = SimpleDirectoryReader("./data").load_data()
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
# per-index
parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(documents)
"""

from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path

md_docs = FlatReader().load_data(Path("./data/cp_handbook.json"))

parser = SimpleFileNodeParser()
md_nodes = parser.get_nodes_from_documents(md_docs)