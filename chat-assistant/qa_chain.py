import pickle
from queue import Queue

from langchain import LLMChain, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.memory import ConversationSummaryBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.faiss import FAISS


from prompts.qa_prompt import QA_PROMPT



class StreamingQueueCallbackHandler(BaseCallbackHandler):
    def __init__(self):
    # def __init__(self, q: Queue):
        # self.q = q
        self.q = Queue(10)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.q.put(None)



class MainChain:

    def __init__(self):
        """Initializes the MainQAChain.

        Args:
            embedder: An instance of Embedder. Should be the general background knowledge.
            callbacks: A list of async callback handlers.
        """
        streaming_callback = StreamingQueueCallbackHandler()
        self.streaming_buffer = streaming_callback.q
        
        self.openai_model = 'gpt-4'
        self.memory_summary_model = 'gpt-3.5-turbo'
        self.qa_chain = self._init_qa_chain([streaming_callback])
        self.ref_doc_retriever = self._load_retriever('vectorstore.pkl')


    def _load_retriever(self, retriever_path):
        with open(retriever_path, "rb") as f:
            retriever = pickle.load(f)
        # retriever = VectorStoreRetriever(vectorstore=vectorstore)
        return retriever

    def _init_qa_chain(self, callbacks=None) -> LLMChain:
        """Initializes the QA chain.

        Args:
            callbacks: A list of async callback handlers.

        Returns:
            The QA chain.
        """
        chat = ChatOpenAI(model_name=self.openai_model, streaming=True, callbacks=callbacks,
                            temperature=0)
        template = QA_PROMPT

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
        memory = ConversationSummaryBufferMemory(max_token_limit=2000, input_key='human_input',
                                                    memory_key="chat_history",
                                                    return_messages=True,
                                                    llm=ChatOpenAI(temperature=0, model_name=self.memory_summary_model))

        chain = LLMChain(llm=chat, prompt=chat_prompt, memory=memory)
        return chain


    def __call__(self, query: str) -> str:
        """Asks a question and gets the answer.

        Args:
            query: The question to ask.

        Returns:
            Answer
        """
        # Reference Document retrieval
        ref_doc = self.ref_doc_retriever.similarity_search(query, k=4)
        ref_doc = '\n===\n'.join(i.page_content for i in ref_doc)

        # Source Code retrieval TODO
        # ref_doc = self.retriever.similarity_search(query, k=4)
        # ref_doc = '\n===\n'.join(i.page_content for i in ref_doc)
        source_code = ''


        result = self.qa_chain.predict(human_input=query, ref_doc=ref_doc, source_code=source_code)
        return result