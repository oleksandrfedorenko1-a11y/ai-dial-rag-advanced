from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )

    def process_text_file(self, file_name: str, chunk_size: int, overlap: int, dimensions: int, truncate: bool = False):
        """
        Process text file: chunk it, generate embeddings, and store in database.
        
        Args:
            file_name: Path to the text file to process
            chunk_size: Size of each text chunk
            overlap: Overlap between consecutive chunks
            dimensions: Embedding dimensions
            truncate: Whether to truncate the vectors table before inserting
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Truncate table if needed
            if truncate:
                cursor.execute("TRUNCATE TABLE vectors RESTART IDENTITY")
            
            # Load content from file
            with open(file_name, 'r') as f:
                content = f.read()
            
            # Generate chunks
            chunks = chunk_text(content, chunk_size, overlap)
            
            # Generate embeddings
            embeddings_dict = self.embeddings_client.get_embeddings(chunks)
            
            # Insert embeddings and chunks to DB
            for index, chunk_text_content in enumerate(chunks):
                embedding = embeddings_dict[index]
                embedding_str = str(embedding)
                
                cursor.execute(
                    """
                    INSERT INTO vectors (document_name, text, embedding) 
                    VALUES (%s, %s, %s::vector)
                    """,
                    (file_name, chunk_text_content, embedding_str)
                )
            
            conn.commit()
            cursor.close()
        finally:
            conn.close()




    def search(self, search_mode: SearchMode, user_request: str, top_k: int, min_score_threshold: float, dimensions: int) -> list[dict]:
        """
        Search for relevant context based on user request using vector similarity.
        
        Args:
            search_mode: SearchMode.EUCLIDIAN_DISTANCE or SearchMode.COSINE_DISTANCE
            user_request: The user's search query
            top_k: Number of top results to return
            min_score_threshold: Minimum distance threshold for results
            dimensions: Embedding dimensions
            
        Returns:
            List of relevant documents with text and distance score
        """
        # Generate embeddings from user request
        request_embeddings = self.embeddings_client.get_embeddings([user_request])
        request_embedding = request_embeddings[0]
        embedding_str = str(request_embedding)
        
        # Select appropriate distance operator
        distance_operator = "<->" if search_mode == SearchMode.EUCLIDIAN_DISTANCE else "<=>"
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search in DB with distance filtering
            query = f"""
                SELECT 
                    text, 
                    embedding {distance_operator} %s::vector AS distance
                FROM vectors
                WHERE embedding {distance_operator} %s::vector < %s
                ORDER BY distance ASC
                LIMIT %s
            """
            
            cursor.execute(query, (embedding_str, embedding_str, min_score_threshold, top_k))
            results = cursor.fetchall()
            cursor.close()
            
            return results
        finally:
            conn.close()

