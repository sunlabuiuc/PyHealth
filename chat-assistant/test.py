import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = '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'

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
memory_chain = ConversationSummaryBufferMemory(max_token_limit=10, llm=llm, return_messages=True)

memory_chain.save_context({"input": "hi"}, {"output": "whats up"})

messages = memory_chain.chat_memory.messages
print(messages)
previous_summary = "hi"
res = memory_chain.predict_new_summary(messages, previous_summary)
print(res)