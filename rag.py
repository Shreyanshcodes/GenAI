from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter,SentenceSplitter
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

#d is dimension of the embedding model to create a faiss index
d = 384
faiss_index = faiss.IndexFlatL2(d)
#Embedding Model definition
embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

#Defining Vector store
vector_store = FaissVectorStore(faiss_index=faiss_index)

#creation of ingestion pipeline this is created using custom configurations
#pipeline persit: caching pipeline for saving time, use cache.clear() to delete the cache if size becomes too big

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=200, chunk_overlap=0),
        embed_model
    
    ], 
    vector_store = vector_store
)
pipeline.persist("./persist_storage")
pipeline.load("./persist_storage")

#
nodes = pipeline.run(
    documents = SimpleDirectoryReader("data").load_data()
)
print(f"Ingested {len(nodes)} Nodes")
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

#creation of index
index = VectorStoreIndex(nodes)


