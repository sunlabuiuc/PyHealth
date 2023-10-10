import os
import time
import random
import threading
from typing import Optional, Tuple
from threading import Lock
import gradio as gr

from qa_chain import MainChain
from prompts.introduction_prompt import AI_INTRO


# with open('OPENAI_API_KEY.txt', 'r') as f:
#     os.environ["OPENAI_API_KEY"] = f.read().strip()

# os.environ["OPENAI_API_KEY"] = 'sk-'
# os.environ["LOG_PATH"] = 'logs'
# os.environ["CORPUS_PATH"] = 'corpus'


CSS = """
    .contain { display: flex; flex-direction: column; }
    .gradio-container { height: 100vh !important; }
    #component-0 { height: 100%; }
    #chatbot { flex-grow: 1; overflow: auto;}
    footer {visibility: hidden}
    """


import logging
from logging.handlers import TimedRotatingFileHandler

# logger configuration
logger = logging.getLogger('chatbot_log')
logger.setLevel(logging.INFO)
# timed rotating file handler
log_path = os.path.join(os.environ['LOG_PATH'], 'chatbot.log')
handler = TimedRotatingFileHandler(log_path, when='midnight', interval=1, backupCount=100)
handler.setLevel(logging.INFO)
# logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(handler)


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

        # self.chain = get_basic_qa_chain()
        self.chain = MainChain()

    def __call__(
        # self, inp: str, history: Optional[Tuple[str, str]], summarized_history: str, result_dict: dict
        self, inp: str, summarized_history: str, result_dict: dict
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            # history = history or []
            # Run chain and append input.
            output, summarized_history = self.chain.run(inp, summarized_history, result_dict)
            result_dict['summarized_history'] = summarized_history
            # output = self.chain(inp, history)
            logger.info('CONVERSATION START')
            logger.info('[User Input] ' + inp)
            logger.info('[AI Output] ' + output)
            logger.info('[Chat Summary] ' + summarized_history)
            logger.info('CONVERSATION END')
            # history.append([inp, output])
        except Exception as e:
            # raise e
            logger.error('[Chatbot Error] ' + str(e))
        finally:
            self.lock.release()
            # pass

        # return history, summarized_history


if __name__ == "__main__":
    
    chat = ChatWrapper()
    with gr.Blocks(theme="default", css=CSS, title="PyHealthChat") as block:
        with gr.Row():
            gr.Markdown(
                "<div><img src='https://raw.githubusercontent.com/sunlabuiuc/PyHealth/master/docs/_static/pyhealth_logos/pyhealth-logo.png' width=140>")
            gr.Markdown(
                "<h1><center style='padding: 25px 0; border: 3px;'>PyHealthChat</center></h1>")
            gr.Markdown(
                "<h3 align=right style='padding: 25px 0'><a href='https://pyhealth.readthedocs.io/en/latest/'> back to docs ></a></h3>")
            # gr.Markdown(
            #     """
            #         <p style="display:inline-block;">
            #             <img src="https://pyhealth.readthedocs.io/en/latest/_static/pyhealth-logo.png" width=140>
            #             <h1><center>PyHealthChat</center></h1>
            #         </p>
            #     """
            # )
        chatbot = gr.Chatbot(value=[[None, AI_INTRO]], elem_id="chatbot")
        # session state
        user_summarized_history = gr.State(value='AI: '+AI_INTRO)

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

        def bot(history, summarized_history):
            user_message = history[-1][0]
            history[-1][1] = ""

            result_dict = {}
            t = threading.Thread(target=chat, args=(user_message, summarized_history, result_dict))
            t.start()
            new_token = chat.chain.streaming_buffer.get()
            while new_token is not None:
                history[-1][1] += new_token
                new_token = chat.chain.streaming_buffer.get()
                yield history, None
            t.join()
            return history, result_dict['summarized_history']


        message.submit(user, [message, chatbot], [message, chatbot], queue=False).then(
            bot, [chatbot, user_summarized_history], [chatbot, user_summarized_history]
        )
        # clear.click(lambda: None, None, chatbot, queue=False)

    block.queue()
    block.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        # favicon_path="https://raw.githubusercontent.com/sunlabuiuc/PyHealth/master/docs/_static/pyhealth_logos/pyhealth-logo.png",
    )
