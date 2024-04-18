import os
from getpass import getpass
from haystack import Pipeline
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret

# ANSI escape codes for colors
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

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
    ChatMessage(content="You are Socrates, engaging in a deep discussion with an interlocutor about the nature of justice. Begin the conversation by asking a thought-provoking question.", role=ChatRole.SYSTEM, name="System")
]

# Starting the rounds of discussion
current_speaker = "Socrates"

for _ in range(6):  # Eight rounds of discussion for illustration
    if current_speaker == "Socrates":
        prompt = f"You are Socrates. Respond to this without including 'Socrates:' in your response: {messages[-1].content}"
        color = GREEN
    else:
        prompt = f"You are the interlocutor. Respond to Socrates' question or statement without including 'Interlocutor:' in your response: {messages[-1].content}"
        color = BLUE
    
    user_message = ChatMessage(content=prompt, role=ChatRole.USER, name=current_speaker)
    messages.append(user_message)

    response = pipe.run(data={"prompt_builder": {"prompt_source": messages, "template_variables": {"0":"0"}}})
    last_reply = response['llm']['replies'][-1].content
    print("=" * 80)
    print(f"{color}{current_speaker}{RESET}: {last_reply}")

    # Ensure response is complete; if not, handle accordingly (not shown here)
    # Switch to the next speaker
    current_speaker = "Interlocutor" if current_speaker == "Socrates" else "Socrates"

    # Append the last reply to keep the context for the next response
    messages.append(ChatMessage(content=last_reply, role=ChatRole.ASSISTANT, name=current_speaker))