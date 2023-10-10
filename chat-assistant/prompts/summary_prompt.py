SUMMARY_PROMPT_TEMPLATE = '''You are an AI who is responsible for summarizing chat history between one another AI and the user.

I will provide you with the summary of previous chat history and new message in the current round of conversation. Then you need to generate new summary based on these information.

Summary of Previous Chat History: {previous_summary}
New Message: {new_message}

The length of your new summary is limited to {summary_token_limitation} tokens.

Now, give the new summary of chat history:'''