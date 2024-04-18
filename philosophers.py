import os
from getpass import getpass
from haystack import Pipeline
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret

# Environment setup for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")

# Initialize the chat prompt builder and the chat generator
prompt_builder = DynamicChatPromptBuilder()
llm = OpenAIChatGenerator(api_key=Secret.from_token(os.environ["OPENAI_API_KEY"]), model="gpt-4-turbo")

# Create a pipeline
pipe = Pipeline()
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", llm)
pipe.connect("prompt_builder.prompt", "llm.messages")

# Initial system message to start the discussion
messages = [
    ChatMessage(content="You are ancient Greek philosophers engaging in a deep discussion about the nature of justice. Begin the conversation.", role=ChatRole.SYSTEM, name="System")
]

# Starting the rounds of discussion
current_philosopher = 1

for _ in range(6):  # Six rounds of discussion for illustration
    prompt = f"You are Philosopher {current_philosopher}. Respond to this without including 'Philosopher {current_philosopher}:' in your response: {messages[-1].content}" if len(messages) > 1 else "You are Philosopher 1. Share your initial thoughts on love without including 'Philosopher 1:' in your response."
    user_message = ChatMessage(content=prompt, role=ChatRole.USER, name=f"Philosopher{current_philosopher}")
    messages.append(user_message)

    response = pipe.run(data={"prompt_builder": {"prompt_source": messages, "template_variables": {"0":"0"}}})
    last_reply = response['llm']['replies'][-1].content
    print(f"Philosopher {current_philosopher}: {last_reply}\n")

    # Ensure response is complete; if not, handle accordingly (not shown here)
    # Switch to the next philosopher
    current_philosopher = 2 if current_philosopher == 1 else 1

    # Append the last reply to keep the context for the next response
    messages.append(ChatMessage(content=last_reply, role=ChatRole.ASSISTANT, name=f"Philosopher{3 - current_philosopher}"))