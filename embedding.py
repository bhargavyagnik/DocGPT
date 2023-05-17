import cohere
import os
from dotenv import load_dotenv
import numpy as np


load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEYS"))

test_sentences = ["This is sentence one", "This is sentence 2"]


embeds = co.embed(
    texts=test_sentences,
    model='small',
    truncate='LEFT'
).embeddings


shape = np.array(embeds).shape
print(shape)