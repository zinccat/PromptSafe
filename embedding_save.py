import openai
import numpy as np
import os

def save_embedding(output, path, name, embedding_model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=output,
        model=embedding_model,
        
    )
    output_embedding = np.array(response["data"][0]["embedding"])
    # save to disk
    np.save(os.path.join(path, "{}.npy".format(name)), output_embedding)
    
if __name__ == '__main__':
    path = './'
    name = 'embedding'
    save_embedding("Revise an academic paragraph for clarity and conciseness without changing its meaning.", path, name)
