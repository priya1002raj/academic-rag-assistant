import os
from dotenv import load_dotenv

# LangChain community imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Core LangChain modules
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from langchain_core.output_parsers import StrOutputParser

from groq import Groq

load_dotenv()


# ----------------------------
# Custom Groq LLM Wrapper
# ----------------------------
class GroqLLM(LLM):
    client: Groq = None
    model_name: str = "llama-3.1-8b-instant"

    def __init__(self, model_name="llama-3.1-8b-instant"):
        super().__init__()
        self.model_name = model_name
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    @property
    def _llm_type(self):
        return "groq_llm"

    def _call(self, prompt, stop=None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content




# ----------------------------
# Load FAISS Vector Store
# ----------------------------
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ----------------------------
# Prompt Template
# ----------------------------
prompt_template = """
You are an academic tutor. Answer the question using ONLY the context.
If the answer does not exist in the context, say:
"I cannot find this in the provided material."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


# ----------------------------
# FULL RAG CHAIN (LCEL)
# ----------------------------
def get_rag_chain():
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = GroqLLM()
    parser = StrOutputParser()

    # LCEL RAG Pipeline
    def rag_pipeline(question):
        # NEW API (invoke)
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        chain_input = {
            "context": context,
            "question": question,
        }

        chain = prompt | llm | parser
        result = chain.invoke(chain_input)

        return result, retrieved_docs

    return rag_pipeline

