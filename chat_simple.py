# ========================================================================
#   Imports
# ========================================================================

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent


# ========================================================================
#   Constants
# ========================================================================

# Model configurations
MODEL_CONFIGS = {
    "openai": {
        "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
        "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
        "gpt-4": {"provider": "openai", "model": "gpt-4"},
        "gpt-3.5-turbo": {"provider": "openai", "model": "gpt-3.5-turbo"},
    },
    "anthropic": {
        "opus": {"provider": "anthropic", "model": "claude-3-opus-20240229"},
        "sonnet": {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
    },
    "gemini": {
        "gemini-2.5-pro": {"provider": "google", "model": "gemini-2.5-pro"},
        "gemini-2.5-flash": {"provider": "google", "model": "gemini-2.5-flash"}
    },
    "mistral": {
        "mistral-small-2506": {"provider": "mistral", "model": "mistral-small-2506"},
        "mistral-medium-2506": {"provider": "mistral", "model": "mistral-medium-2506"},
    },
    "meta": {
        "llama-70b": {"provider": "meta", "model": "meta-llama-3-70b-instruct"},
        "llama-8b": {"provider": "meta", "model": "meta-llama-3-8b-instruct"},
    }
}

# Default configuration
DEFAULT_MODEL = MODEL_CONFIGS["openai"]["gpt-3.5-turbo"]
DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI assistant."
LOG_DIR = "logs"


# ========================================================================
#   Logging Setup
# ========================================================================

def setup_logging(log_dir: str = LOG_DIR) -> logging.Logger:
    """
    Set up logging configuration with both file and console output.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate timestamp for unique log filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_path / f"autogen_chat_{timestamp}.log"
    
    # Configure logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    
    # Log the startup
    logger.info("=" * 50)
    logger.info("AutoGen Chat Session Started")
    logger.info(f"Log file: {log_filename}")
    logger.info("=" * 50)
    
    return logger


# ========================================================================
#   Environment Configuration
# ========================================================================

def load_api_keys() -> Dict[str, Optional[str]]:
    """
    Load API keys from environment variables.
    
    Returns:
        Dictionary containing API keys for each provider
    """
    load_dotenv()
    logger.info("Loading environment variables from .env file")
    
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "mistral": os.getenv("MISTRAL_API_KEY"),
        "meta": os.getenv("META_API_KEY")
    }
    
    # Log which API keys were found (without revealing the actual keys)
    for provider, key in api_keys.items():
        if key:
            logger.info(f"{provider.capitalize()} API key found")
        else:
            logger.warning(f"{provider.capitalize()} API key not found")
    
    return api_keys


def validate_api_key(api_keys: Dict[str, Optional[str]], provider: str) -> str:
    """
    Validate that the API key exists for the selected provider.
    
    Args:
        api_keys: Dictionary of API keys
        provider: Selected provider name
        
    Returns:
        The API key for the provider
        
    Raises:
        ValueError: If API key is not found
    """
    api_key = api_keys.get(provider)
    if not api_key:
        logger.error(f"No API key found for provider: {provider}")
        raise ValueError(f"API key not found for {provider}")
    
    logger.info(f"Using {provider.capitalize()} API key")
    return api_key


# ========================================================================
#   Model Configuration
# ========================================================================

def create_llm_config(model_config: Dict[str, str], api_key: str) -> Dict[str, Any]:
    """
    Create the LLM configuration for AutoGen.
    
    Args:
        model_config: Model configuration dictionary
        api_key: API key for the provider
        
    Returns:
        LLM configuration dictionary
    """
    llm_config = {
        "config_list": [
            {
                "api_type": model_config["provider"],
                "model": model_config["model"],
                "api_key": api_key,
            }
        ]
    }
    
    logger.info(f"LLM configuration created for {model_config['provider']} - {model_config['model']}")
    return llm_config


# ========================================================================
#   Agent Creation
# ========================================================================

def create_agents(llm_config: Dict[str, Any], 
                 system_message: str = DEFAULT_SYSTEM_MESSAGE) -> Tuple[AssistantAgent, UserProxyAgent]:
    """
    Create and configure AutoGen agents.
    
    Args:
        llm_config: LLM configuration dictionary
        system_message: System message for the assistant
        
    Returns:
        Tuple of (assistant, user_proxy) agents
    """
    logger.info("Initializing agents...")
    
    # Create assistant agent
    assistant = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
        human_input_mode="NEVER",
        system_message=system_message
    )
    logger.info("Assistant agent created")
    
    # Create user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False  # Explicitly disable code execution
    )
    logger.info("User proxy agent created")
    
    return assistant, user_proxy


# ========================================================================
#   Chat Execution
# ========================================================================

def run_chat(assistant: AssistantAgent, 
             user_proxy: UserProxyAgent, 
             message: str) -> str:
    """
    Execute a single-turn chat between user and assistant.
    
    Args:
        assistant: The assistant agent
        user_proxy: The user proxy agent
        message: User's message
        
    Returns:
        Assistant's response
        
    Raises:
        Exception: If chat execution fails
    """
    try:
        logger.info(f"User message: {message}")
        logger.info("Initiating chat between user_proxy and assistant")
        
        # Initiate the chat
        response = user_proxy.initiate_chat(
            recipient=assistant,
            message=message,
            clear_history=True  # Clear history for each new chat
        )
        
        logger.info("Chat completed successfully")
        
        # Extract the assistant's reply
        assistant_reply = response.chat_history[-1]['content']
        
        # Log reply preview
        preview_length = 100
        if len(assistant_reply) > preview_length:
            logger.info(f"Assistant reply: {assistant_reply[:preview_length]}...")
        else:
            logger.info(f"Assistant reply: {assistant_reply}")
        
        logger.info(f"Total messages in chat: {len(response.chat_history)}")
        
        return assistant_reply
        
    except Exception as e:
        logger.error(f"An error occurred during chat: {str(e)}", exc_info=True)
        raise


# ========================================================================
#   Main Application
# ========================================================================

def main() -> None:
    """
    Main function that orchestrates the chat application.
    """
    global logger
    
    try:
        # Initialize logging
        logger = setup_logging()
        
        # Load API keys
        api_keys = load_api_keys()
        
        # UPDATE: Change this to use different models
        selected_model = DEFAULT_MODEL
        logger.info(f"Selected model: {selected_model['provider']} - {selected_model['model']}")
        
        # Validate API key
        api_key = validate_api_key(api_keys, selected_model["provider"])
        
        # Create LLM configuration
        llm_config = create_llm_config(selected_model, api_key)
        
        # Create agents
        assistant, user_proxy = create_agents(llm_config)
        
        # UPDATE: User's message
        message = "Summarize what a neural network is in one paragraph."
        
        # Run the chat
        response = run_chat(assistant, user_proxy, message)
        
        # Print the response
        print("\n" + "="*50)
        print("Assistant's Response:")
        print("="*50)
        print(response)
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)
        
    finally:
        logger.info("="*50)
        logger.info("AutoGen Chat Session Ended")
        logger.info("="*50)


# ========================================================================
#   Entry Point
# ========================================================================

if __name__ == "__main__":
    main()