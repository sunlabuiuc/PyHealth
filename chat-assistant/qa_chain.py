import pickle
import os
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


from prompts.qa_prompt import QA_PROMPT_TEMPLATE
from prompts.summary_prompt import SUMMARY_PROMPT_TEMPLATE
from prompts.introduction_prompt import USER_INTRO, AI_INTRO



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
        self.memory_chain = self._init_memory_chain()
        self.corpus_path = os.environ['CORPUS_PATH']
        self.ref_doc_retriever = self._load_retriever(os.path.join(self.corpus_path, 'pyhealth-text.pkl'))
        self.source_code_retriever = self._load_retriever(os.path.join(self.corpus_path, 'pyhealth-code.pkl'))
        self.summary_token_limitation = 2000
        self.topk = 4


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
        llm = ChatOpenAI(model_name=self.openai_model, streaming=True, callbacks=callbacks,
                            temperature=0)
        template = QA_PROMPT_TEMPLATE
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
        chain = LLMChain(llm=llm, prompt=chat_prompt)
        return chain
    
    def _init_memory_chain(self) -> LLMChain:
        llm = ChatOpenAI(model_name=self.memory_summary_model, temperature=0)
        prompt = PromptTemplate(input_variables=['new_message', 'previous_summary', 'summary_token_limitation'], template=SUMMARY_PROMPT_TEMPLATE)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain


    def run(self, query: str, summarized_history: str, result_dict: dict) -> str:
        """Asks a question and gets the answer.

        Args:
            query: The question to ask.

        Returns:
            Answer
        """
        # Reference Document retrieval
        ref_doc = self.ref_doc_retriever.similarity_search(query, k=self.topk)
        ref_doc = '\n===\n'.join(i.page_content for i in ref_doc)

        # Source Code retrieval
        source_code = self.source_code_retriever.similarity_search(query, k=self.topk)
        source_code = '\n===\n'.join(i.page_content for i in source_code)
        
        result = self.qa_chain.predict(human_input=query, chat_history=summarized_history, ref_doc=ref_doc, source_code=source_code)

        # summarize new history
        new_message = 'Human:\n' + query + '\nAI:\n' + result + '\n'
        summarized_history = self.memory_chain.predict(new_message=new_message, previous_summary=summarized_history, summary_token_limitation=str(self.summary_token_limitation))

        return result, summarized_history