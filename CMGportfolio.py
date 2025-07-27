import pandas as pd
import uuid
import os

class Portfolio:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        self.data = pd.read_csv(file_path)

        required_columns = {"Techstack", "Links"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        base_dir = os.getcwd()
        self.vectorstore_path = os.path.join(base_dir, "vectorstore")
        os.makedirs(self.vectorstore_path, exist_ok=True)

        # Initialize chroma client and collection as None
        self.chroma_client = None
        self.collection = None

    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection only when needed."""
        if self.chroma_client is None:
            try:
                import chromadb
                self.chroma_client = chromadb.PersistentClient(path=self.vectorstore_path)
                self.collection = self.chroma_client.get_or_create_collection(name="portfolio")
                self.load_portfolio()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize chroma client: {e}")

    def load_portfolio(self):
        if self.collection is not None and not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        # Initialize ChromaDB only when query_links is called
        self._initialize_chroma()
        return self.collection.query(query_texts=skills, n_results=2).get('metadatas', [])