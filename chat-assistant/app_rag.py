import os
import time
import random
import threading
from typing import Optional, Tuple
from threading import Lock
import gradio as gr
from env import OPENAI_API_KEY


from qa_chain import MainChain
from prompts.introduction_prompt import AI_INTRO

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

CSS = """
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto;}
    """


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

        # self.chain = get_basic_qa_chain()
        self.chain = MainChain()

    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            # history = history or []
            # Run chain and append input.
            output = self.chain(inp)
            # history.append([inp, output])
        except Exception as e:
            raise e
        finally:
            self.lock.release()
            # pass
        return history


if __name__ == "__main__":

    chat = ChatWrapper()
    with gr.Blocks(theme="default", css=CSS) as block:
        with gr.Row():
            gr.Markdown(
                "<h1><center>PyHealth Assistant</center></h1> <h3><a href='https://pyhealth.readthedocs.io/en/latest/'>< back to docs</a></h3>")
        chatbot = gr.Chatbot(value=[[None, AI_INTRO]], elem_id="chatbot")

        with gr.Row():
            message = gr.Textbox(
                label="What's your question about PyHealth?",
                placeholder="Type in your question here...",
                lines=1,
            )

        # clear = gr.Button("Clear")
        

        gr.Examples(
            examples=[
                "How can PyHealth help you?",
                "Where is the API documentation of PyHealth?",
                "Can you show me some simple PyHealth tutorials?",
            ],
            inputs=message,
        )

        # state = gr.State()
        # message.submit(chat, inputs=[message, state], outputs=[chatbot, state])

        # # https://discuss.huggingface.co/t/unable-to-clear-input-after-submit/33543/12
        # message.submit(lambda x: gr.update(value=""),
        #                [state], [message], queue=False)

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            user_message = history[-1][0]
            history[-1][1] = ""

            t = threading.Thread(target=chat, args=(user_message, history))
            t.start()
            new_token = chat.chain.streaming_buffer.get()
            while new_token is not None:
                history[-1][1] += new_token
                new_token = chat.chain.streaming_buffer.get()
                yield history
            
            t.join()
        message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        # clear.click(lambda: None, None, chatbot, queue=False)

    block.queue()
    block.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0"
    )
