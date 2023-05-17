from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer :"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)




from dotenv import load_dotenv
load_dotenv()


loader = PyPDFLoader("resume.pdf")
pages = loader.load_and_split()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

embeddings = CohereEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":5})
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=Cohere(), chain_type="stuff",retriever=retriever, return_source_documents=True,chain_type_kwargs=chain_type_kwargs)

print("Starting Chat, (q) to Exit :")
while True:
    print("test")
    question = input("Chat : ")
    print(qa({'query':question}))
