import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        """
        Initialize the DIAL Embeddings Client.
        
        Args:
            deployment_name: The name of the embedding model deployment
            api_key: The API key for authentication
        """
        self.deployment_name = deployment_name
        self.api_key = api_key
    
    def get_embeddings(self, texts: list) -> dict:
        """
        Generate embeddings for input list.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            Dict with indexed embeddings (key is index from input list, value is vector list)
        """
        url = DIAL_EMBEDDINGS.format(model=self.deployment_name)
        
        headers = {
            'Api-Key': f'{self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'input': texts,
            'model': self.deployment_name
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        
        # Convert response to indexed embeddings dict
        embeddings_dict = {}
        for item in result['data']:
            index = item['index']
            embedding = item['embedding']
            embeddings_dict[index] = embedding
        
        return embeddings_dict


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
