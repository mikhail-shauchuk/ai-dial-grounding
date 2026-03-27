import asyncio
from typing import Any
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `vector_based_grounding.png` to see the flow of app


SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about user information.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}

##USER QUESTION: 
{query}"""


def format_user_document(user: dict[str, Any]) -> str:
    context_str = "User:\n"
    for key, value in user.items():
        context_str += f"  {key}: {value}\n"
    return context_str


class UserRAG:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = None

    async def __aenter__(self):
        print("🔎 Loading all users...")
        all_users = UserClient().get_all_users()
        documents = [Document(page_content=format_user_document(user)) for user in all_users]

        print(f"↗️ Creating embeddings and vectorstore for {len(documents)} documents...")
        self.vectorstore = await self._create_vectorstore_with_batching(documents)
        print("✅ Vectorstore is ready.")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def _create_vectorstore_with_batching(self, documents: list[Document], batch_size: int = 100):
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        batch_tasks = [FAISS.afrom_documents(batch, self.embeddings) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        final_vectorstore = None
        for batch_vectorstore in batch_results:
            if isinstance(batch_vectorstore, Exception):
                continue
            if final_vectorstore is None:
                final_vectorstore = batch_vectorstore
            else:
                final_vectorstore.merge_from(batch_vectorstore)

        if final_vectorstore is None:
            raise ValueError("Failed to create vectorstore: all batches failed.")
        return final_vectorstore

    async def retrieve_context(self, query: str, k: int = 10, score: float = 0.1) -> str:
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        context_parts = []
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        return response.content


async def main():
    embeddings = AzureOpenAIEmbeddings(
        deployment='text-embedding-3-small-1',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
    )

    llm_client = AzureChatOpenAI(
        temperature=0.0,
        azure_deployment='gpt-4o',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        api_version="2024-02-01",
    )

    async with UserRAG(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need user emails that filled with hiking and psychology")
        print(" - Who is John?")
        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            context = await rag.retrieve_context(user_question)
            augmented_prompt = rag.augment_prompt(user_question, context)
            answer = rag.generate_answer(augmented_prompt)
            print(answer)


asyncio.run(main())

# The problems with Vector based Grounding approach are:
#   - In current solution we fetched all users once, prepared Vector store (Embed takes money) but we didn't play
#     around the point that new users added and deleted every 5 minutes. (Actually, it can be fixed, we can create once
#     Vector store and with new request we will fetch all the users, compare new and deleted with version in Vector
#     store and delete the data about deleted users and add new users).
#   - Limit with top_k (we can set up to 100, but what if the real number of similarity search 100+?)
#   - With some requests works not so perfectly. (Here we can play and add extra chain with LLM that will refactor the
#     user question in a way that will help for Vector search, but it is also not okay in the point that we have
#     changed original user question).
#   - Need to play with balance between top_k and score_threshold
# Benefits are:
#   - Similarity search by context
#   - Any input can be used for search
#   - Costs reduce