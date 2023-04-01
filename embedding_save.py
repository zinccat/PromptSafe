import openai
import numpy as np

def embedding(output, embedding_model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=output,
        model=embedding_model
    )
    output_embedding = np.array(response["data"][0]["embedding"])
    # save to disk
    np.save("embedding.npy", output_embedding)
    

embedding("Revise an academic paragraph for clarity and conciseness without changing its meaning.")
