# ========================================================================
#   Imports
# ========================================================================

# multi_agent_example.py
import os
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent


# ========================================================================
#   Environment Variables
# ========================================================================

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
meta_api_key = os.getenv("META_API_KEY")  


# ========================================================================
#   Model Presets
# ========================================================================

# Assign specific models to variables for reuse without prompting
openai_gpt_4o = {"provider": "openai", "model": "gpt-4o"}
openai_gpt_4o_mini = {"provider": "openai", "model": "gpt-4o-mini"}
openai_gpt_4 = {"provider": "openai", "model": "gpt-4"}
openai_gpt_35 = {"provider": "openai", "model": "gpt-3.5-turbo"}
anthropic_opus = {"provider": "anthropic", "model": "claude-3-opus-20240229"}
anthropic_sonnet = {"provider": "anthropic", "model": "claude-3-sonnet-20240229"}
meta_llama_70b = {"provider": "meta", "model": "meta-llama-3-70b-instruct"}
meta_llama_8b = {"provider": "meta", "model": "meta-llama-3-8b-instruct"}

# UPDATE: Choose the model to use by assigning one of the above to selected_entry
selected_model = openai_gpt_35  


# ========================================================================
#   LLM Configuration
# ========================================================================

llm_config = {
    "config_list": [{
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }],
}


# ========================================================================
#   Agent Initialization
# ========================================================================

# Create a writer agent
writer = AssistantAgent(
    name="Writer",
    system_message="You are a creative writer who writes short stories.",
    llm_config=llm_config,
)

# Create a critic agent
critic = AssistantAgent(
    name="Critic",
    system_message="You are a literary critic who provides constructive feedback.",
    llm_config=llm_config,
)

# User proxy to coordinate
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,               # Allow back-and-forth
    code_execution_config=False,
)


# ========================================================================
#   Main Execution
# ========================================================================

def main() -> None:
    """
    
    Main function that launches the chat.
    The user asks a question, the assistant responds, and the result is printed.
    
    """
    # UPDATE: User's message
    writer_message = "Write a 3-sentence story about a robot learning to paint."

    # Initiate the chat
    response = user_proxy.initiate_chat(        
        recipient=writer,                    # Specifies who receives messages from the user
        message=writer_message                         # Message sent to the LLM
    )  
    
    
    critic_message = f"Please critique this story: {writer.last_message()['content']}"
    
    # Get the critic's feedback
    user_proxy.initiate_chat(
        recipient=critic,
        message=critic_message
    )

    # Print the assistant’s reply to the terminal
    print(response.chat_history[-1]['content']) 


if __name__ == "__main__":
    main()  