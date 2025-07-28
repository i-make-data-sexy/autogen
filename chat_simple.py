# ========================================================================
#   Imports
# ========================================================================

import os                  
from dotenv import load_dotenv     
from autogen import AssistantAgent, UserProxyAgent


# ========================================================================
#   Environment Variables
# ========================================================================

 # Load variables from .env 
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

# Choose the correct API key based on provider
if selected_model["provider"] == "openai":
    selected_api_key = openai_api_key
elif selected_model["provider"] == "anthropic":
    selected_api_key = anthropic_api_key
else:
    selected_api_key = meta_api_key

# Set which model to query
llm_config = {
    "config_list": [
        {
            "api_type": selected_model["provider"],
            "model": selected_model["model"],
            "api_key": selected_api_key,
        }
    ]
}


# ========================================================================
#   Agent Initialization
# ========================================================================

# LLM agent that will generate responses
assistant = AssistantAgent(
    name="assistant", 
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message="You are a helpful AI assistant."
)

# User who will send the prompts
user_proxy = UserProxyAgent(
    name="user_proxy",                                  # Chat user        
    human_input_mode="NEVER",                           # Don"t ask user to respond to anything in the terminal
    max_consecutive_auto_reply=1                        # Only let the assistant reply once
)

# ========================================================================
#   Main Execution
# ========================================================================

def main() -> None:
    """
    
    Main function that launches a single-turn chat.
    The user asks a question, the assistant responds, and the result is printed.
    
    """
    # UPDATE: User's message
    message = "Summarize what a neural network is in one paragraph."

    # Initiate the chat
    response = user_proxy.initiate_chat(        
        recipient=assistant,                    # Specifies who receives messages from the user
        message=message                         # Message sent to the LLM
    )    

    # Print the assistant’s reply to the terminal
    print(response.chat_history[-1]['content']) 


if __name__ == "__main__":
    main()  