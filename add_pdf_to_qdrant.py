from pathlib import Path
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.writers import DocumentWriter

# Define the index name and document path as variables
collection_name = "SIBS"
document_path = "sibs.pdf"

pipeline = Pipeline()

# Components
converter = PyPDFToDocument()
cleaner = DocumentCleaner()
splitter = DocumentSplitter(split_by="sentence", split_length=10)
document_embedder = OpenAIDocumentEmbedder(api_key=Secret.from_env_var("OPENAI_API_KEY"))
document_store = QdrantDocumentStore(
    url="localhost",
    index=collection_name,
    embedding_dim=1536,
    recreate_index=True,
)
writer = DocumentWriter(document_store=document_store)

# Add components to pipeline
pipeline.add_component("converter", converter)
pipeline.add_component("cleaner", cleaner)
pipeline.add_component("splitter", splitter)
pipeline.add_component("embedder", document_embedder)  
pipeline.add_component("writer", writer)

# Connect components
pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "embedder") 
pipeline.connect("embedder", "writer")

# Run the pipeline
res = pipeline.run({"converter": {"sources": [Path(document_path)]}})

print(res)