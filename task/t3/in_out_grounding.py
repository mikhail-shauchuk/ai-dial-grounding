import asyncio
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from pydantic import SecretStr, BaseModel, Field
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO: Info about app:
# HOBBIES SEARCHING WIZARD
# Before implementation open the `flow.png` to see the flow of app.
# Searches users by hobbies and provides their full info in JSON format:
#   Input: I need people who love to go to mountains
#   Output:
#     rock climbing: [{full user info JSON},...],
#     hiking: [{full user info JSON},...],
#     camping: [{full user info JSON},...]
# ---
# 1. Since we are searching hobbies that persist in `about_me` section - we need to embed only user `id` and `about_me`!
#    It will allow us to reduce context window significantly.
# 2. Pay attention that every 5 minutes in system will be added new users and some will be deleted. We will at the
#    'cold start' add all users for current moment to vectorstor and with each user request we will update vectorstor,
#    we will remove deleted users and add new - it will also resolve the issue with consistency within this 2 services
#    and will reduce costs (we don't need on each user request load vectorstor from scratch and pay for it).
# 3. We ask LLM make NEE (Named Entity Extraction) https://cloud.google.com/discover/what-is-entity-extraction?hl=en
#    and provide response in format:
#    {
#       "{hobby}": [{user_id}, 2, 4, 100...]
#    }
#    It allows us to save significant money on generation, reduce time on generation and eliminate possible
#    hallucinations (corrupted personal info or removed some parts of PII (Personal Identifiable Information)). After
#    generation we also need to make output grounding (fetch full info about user and in the same time check that all
#    presented IDs are correct).
# 4. In response we expect JSON with grouped users by their hobbies.
# ---
# This sample is based on the real solution where one Service provides our Wizard with user request, we fetch all
# required data and then returned back to 1st Service response in JSON format.

SYSTEM_PROMPT = """You are a RAG-powered assistant that groups users by their hobbies.

## Flow:
Step 1: User will ask to search users by their hobbies etc.
Step 2: Will be performed search in the Vector store to find most relevant users.
Step 3: You will be provided with CONTEXT (most relevant users, there will be user ID and information about user), and 
        with USER QUESTION.
Step 4: You group by hobby users that have such hobby and return response according to Response Format

## Response Format:
{format_instructions}
"""

USER_PROMPT = """## CONTEXT:
{context}

## USER QUESTION: 
{query}"""


llm_client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment='gpt-4o',
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version="2024-02-01",
)


class GroupingResult(BaseModel):
    hobby: str = Field(description="Hobby. Example: football, painting, horsing, photography, bird watching...")
    user_ids: list[int] = Field(description="List of user IDs that have hobby requested by user.")


class GroupingResults(BaseModel):
    grouping_results: list[GroupingResult] = Field(description="List matching search results.")


def format_user_document(user: dict[str, Any]) -> str:
    return (
        f"User:\n"
        f"  id: {user.get('id')}\n"
        f"  About user: {user.get('about_me')}\n"
        f"---"
    )


class InputGrounder:
    def __init__(self, embeddings: AzureOpenAIEmbeddings, llm_client: AzureChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.user_client = UserClient()
        self.vectorstore = None

    async def __aenter__(self):
        await self.initialize_vectorstore()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def initialize_vectorstore(self, batch_size: int = 50):
        """Initialize vectorstore with all current users."""
        print("🔍 Loading all users for initial vectorstore...")
        users = self.user_client.get_all_users()
        documents = [Document(id=str(user.get('id')), page_content=format_user_document(user)) for user in users]
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        self.vectorstore = Chroma(
            collection_name="users",
            embedding_function=self.embeddings,
        )
        tasks = [self.vectorstore.aadd_documents(batch) for batch in batches]
        await asyncio.gather(*tasks)

    async def retrieve_context(self, query: str, k: int = 100, score: float = 0.2) -> str:
        """Retrieve context, with optional automatic vectorstore update."""
        await self._update_vectorstore()
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        context_parts = []
        for doc, relevance_score in results:
            context_parts.append(doc.page_content)
            print(f"Retrieved (Score: {relevance_score:.3f}): {doc.page_content}")
        return "\n\n".join(context_parts)

    async def _update_vectorstore(self):
        users = self.user_client.get_all_users()
        vectorstore_data = self.vectorstore.get()
        vectorstore_ids_set = set(str(user_id) for user_id in vectorstore_data.get("ids", []))
        users_dict = {str(user.get('id')): user for user in users}
        users_ids_set = set(users_dict.keys())
        new_user_ids = users_ids_set - vectorstore_ids_set
        ids_to_delete = list(vectorstore_ids_set - users_ids_set)
        if ids_to_delete:
            self.vectorstore.delete(ids=ids_to_delete)
        new_documents = [
            Document(id=user_id, page_content=format_user_document(users_dict[user_id]))
            for user_id in new_user_ids
        ]
        if new_documents:
            await self.vectorstore.aadd_documents(new_documents)

    def augment_prompt(self, query: str, context: str) -> str:
        return USER_PROMPT.format(context=context, query=query)

    def generate_answer(self, augmented_prompt: str) -> GroupingResults:
        parser = PydanticOutputParser(pydantic_object=GroupingResults)
        messages = [
            SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(messages=messages).partial(
            format_instructions=parser.get_format_instructions()
        )
        grouping_results: GroupingResults = (prompt | llm_client | parser).invoke({})
        return grouping_results


class OutputGrounder:
    def __init__(self):
        self.user_client = UserClient()

    async def ground_response(self, grouping_results: GroupingResults):
        for grouping_result in grouping_results.grouping_results:
            print(f"\nHobby: {grouping_result.hobby}")
            users = await self._find_users(grouping_result.user_ids)
            print(users)

    async def _find_users(self, ids: list[int]) -> list[dict[str, Any]]:
        async def safe_get_user(user_id: int) -> Optional[dict[str, Any]]:
            try:
                return await self.user_client.get_user(user_id)
            except Exception as e:
                if "404" in str(e):
                    print(f"User with ID {user_id} is absent (404)")
                    return None
                raise  # Re-raise non-404 errors

        tasks = [safe_get_user(user_id) for user_id in ids]
        results = await asyncio.gather(*tasks)
        return [user for user in results if user is not None]


async def main():
    embeddings = AzureOpenAIEmbeddings(
        deployment='text-embedding-3-small-1',
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY),
        dimensions=384,
        check_embedding_ctx_length=False,
    )
    output_grounder = OutputGrounder()

    async with InputGrounder(embeddings, llm_client) as rag:
        print("Query samples:")
        print(" - I need people who love to go to mountains")
        print(" - Find people who love to watch stars and night sky")
        print(" - I need people to go to fishing together")

        while True:
            user_question = input("> ").strip()
            if user_question.lower() in ['quit', 'exit']:
                break
            context = await rag.retrieve_context(user_question)
            augmented = rag.augment_prompt(user_question, context)
            grouping_results = rag.generate_answer(augmented)
            await output_grounder.ground_response(grouping_results)


if __name__ == "__main__":
    asyncio.run(main())
