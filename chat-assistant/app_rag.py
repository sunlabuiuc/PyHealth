import os
from typing import Optional, Tuple
from threading import Lock
import gradio as gr
from query_data import get_basic_qa_chain, get_qa_with_sources_chain
from constant import OPENAI_API_KEY

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

        # Set OpenAI key
        import openai
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.chain = get_basic_qa_chain()

    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # Run chain and append input.
            output = self.chain({"question": inp})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


if __name__ == "__main__":

    chat = ChatWrapper()
    block = gr.Blocks(theme="default", css=CSS)
    with block:
        with gr.Row():
            gr.Markdown(
                "<h1><center>PyHealth Assistant</center></h1> <h3><a href='https://pyhealth.readthedocs.io/en/latest/'>< back to docs</a></h3>")
        chatbot = gr.Chatbot(elem_id="chatbot")

        with gr.Row():
            message = gr.Textbox(
                label="What's your question about PyHealth?",
                placeholder="Type in your question here...",
                lines=1,
            )

        gr.Examples(
            examples=[
                "How can PyHealth help you?",
                "Where is the API documentation of PyHealth?",
                "Can you show me some simple PyHealth tutorials?",
            ],
            inputs=message,
        )

        state = gr.State()
        message.submit(chat, inputs=[message, state], outputs=[chatbot, state])

        # https://discuss.huggingface.co/t/unable-to-clear-input-after-submit/33543/12
        message.submit(lambda x: gr.update(value=""),
                       [state], [message], queue=False)

    block.launch(
        share=False,
        debug=False,
        server_name="0.0.0.0"
    )
