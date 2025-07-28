# ========================================================================
#   Imports
# ========================================================================

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent


# ========================================================================
#   Constants and Configuration
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
        "sonnet": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022"},
    },
    "google": {
        "gemini-2.0-pro": {"provider": "google", "model": "gemini-2.0-flash-exp"},
        "gemini-2.0-flash": {"provider": "google", "model": "gemini-2.0-flash-exp"}
    },
    "mistral": {
        "mistral-small": {"provider": "mistral", "model": "mistral-small-latest"},
        "mistral-medium": {"provider": "mistral", "model": "mistral-medium-latest"},
    },
    "meta": {
        "llama-70b": {"provider": "meta", "model": "meta-llama-3-70b-instruct"},
        "llama-8b": {"provider": "meta", "model": "meta-llama-3-8b-instruct"},
    }
}

# Default configurations
DEFAULT_WRITER_MODEL = MODEL_CONFIGS["openai"]["gpt-4"]
DEFAULT_CRITIC_MODEL = MODEL_CONFIGS["anthropic"]["sonnet"]  # Using Anthropic for the critic!
LOG_DIR = "logs"

# UPDATE: System messages for agents
WRITER_MESSAGE = """You are a creative writer who excels at crafting 
compelling short stories. You have a talent for vivid imagery, 
engaging characters, and emotional depth. Write concisely but powerfully."""

CRITIC_MESSAGE = """You are a thoughtful literary critic who provides 
constructive feedback. You analyze stories for their strengths and 
areas for improvement, always maintaining a supportive tone while 
being specific and actionable in your suggestions."""

DEFAULT_WRITING_PROMPT = """Write a 3-sentence story about a robot learning to paint. 
The story should be emotionally engaging and have a clear beginning, middle, and end."""

# ========================================================================
#   Data Classes
# ========================================================================

@dataclass
class AgentConfig:
    """Configuration for an AutoGen agent."""
    name: str
    system_message: str
    model_config: Dict[str, str]
    max_consecutive_auto_reply: int = 1


@dataclass
class StoryResult:
    """Container for story writing results."""
    original_story: str
    critique: str
    revised_story: Optional[str] = None
    full_history: List[Dict[str, Any]] = None


# ========================================================================
#   Logging Setup
# ========================================================================

def setup_logging(log_dir: str = LOG_DIR, session_name: str = "multi_agent") -> logging.Logger:
    """
    Set up logging configuration with both file and console output.
    
    Args:
        log_dir: Directory to store log files
        session_name: Name prefix for the log file
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate timestamp for unique log filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_path / f"{session_name}_{timestamp}.log"
    
    # Configure logging
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
    logger.info("=" * 70)
    logger.info(f"Multi-Agent Writing Session Started")
    logger.info(f"Log file: {log_filename}")
    logger.info("=" * 70)
    
    return logger


# ========================================================================
#   Environment and API Key Management
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
        "google": os.getenv("GEMINI_API_KEY"),
        "mistral": os.getenv("MISTRAL_API_KEY"),
        "meta": os.getenv("META_API_KEY")
    }
    
    # Log which API keys were found
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
        raise ValueError(f"API key not found for {provider}. Please check your .env file.")
    
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
    # AutoGen expects different config formats for different providers
    if model_config["provider"] == "anthropic":
        llm_config = {
            "config_list": [
                {
                    "model": model_config["model"],
                    "api_key": api_key,
                    "api_type": "anthropic",
                }
            ]
        }
    else:
        # OpenAI and others
        llm_config = {
            "config_list": [
                {
                    "model": model_config["model"],
                    "api_key": api_key,
                }
            ]
        }
    
    logger.info(f"Created LLM config for {model_config['provider']} - {model_config['model']}")
    return llm_config


# ========================================================================
#   Agent Creation
# ========================================================================

def create_writing_agents(agent_configs: List[AgentConfig], 
                         api_keys: Dict[str, Optional[str]]) -> Dict[str, AssistantAgent]:
    """
    Create all writing agents based on configurations.
    
    Args:
        agent_configs: List of agent configurations
        api_keys: Dictionary of API keys
        
    Returns:
        Dictionary of agent name to agent instance
    """
    agents = {}
    
    for config in agent_configs:
        # Validate API key for this agent's model
        api_key = validate_api_key(api_keys, config.model_config["provider"])
        
        # Create LLM configuration
        llm_config = create_llm_config(config.model_config, api_key)
        
        # Create the agent
        agent = AssistantAgent(
            name=config.name,
            system_message=config.system_message,
            llm_config=llm_config,
            max_consecutive_auto_reply=config.max_consecutive_auto_reply
        )
        
        agents[config.name] = agent
        logger.info(f"Created {config.name} agent using {config.model_config['model']}")
    
    return agents


def create_user_proxy(max_consecutive_auto_reply: int = 3) -> UserProxyAgent:
    """
    Create a user proxy agent to coordinate the conversation.
    
    Args:
        max_consecutive_auto_reply: Maximum number of consecutive auto-replies
        
    Returns:
        Configured UserProxyAgent
    """
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        code_execution_config=False,
    )
    logger.info("Created user proxy agent")
    return user_proxy


# ========================================================================
#   Story Writing Workflow
# ========================================================================

def write_initial_story(user_proxy: UserProxyAgent, 
                       writer: AssistantAgent, 
                       prompt: str) -> str:
    """
    Have the writer create the initial story.
    
    Args:
        user_proxy: User proxy agent
        writer: Writer agent
        prompt: Writing prompt
        
    Returns:
        The generated story
    """
    logger.info(f"Starting story writing with prompt: {prompt}")
    
    response = user_proxy.initiate_chat(
        recipient=writer,
        message=prompt,
        clear_history=True
    )
    
    # Get the actual story from the chat history
    # The story is in the Writer's response, not the last message
    story = response.chat_history[1]["content"]  # Index 1 is the Writer's response
    logger.info(f"Writer completed story ({len(story)} characters)")
    
    return story


def get_critique(user_proxy: UserProxyAgent, 
                 critic: AssistantAgent, 
                 story: str) -> str:
    """
    Have the critic provide feedback on the story.
    
    Args:
        user_proxy: User proxy agent
        critic: Critic agent
        story: The story to critique
        
    Returns:
        The critique
    """
    logger.info("Getting critique of the story")
    
    critique_prompt = f"""Please provide constructive feedback on this story. 
    Consider elements like narrative structure, character development, 
    imagery, and emotional impact. Be specific and helpful.
    
    Story:
    {story}"""
    
    response = user_proxy.initiate_chat(
        recipient=critic,
        message=critique_prompt,
        clear_history=True
    )
    
    # Get the actual critique from the chat history
    critique = response.chat_history[1]["content"]  # Index 1 is the Critic's response
    logger.info(f"Critic provided feedback ({len(critique)} characters)")
    
    return critique


def revise_story(user_proxy: UserProxyAgent, 
                writer: AssistantAgent, 
                original_story: str, 
                critique: str) -> str:
    """
    Have the writer revise the story based on critique.
    
    Args:
        user_proxy: User proxy agent
        writer: Writer agent
        original_story: The original story
        critique: The critique to address
        
    Returns:
        The revised story
    """
    logger.info("Getting revised story based on critique")
    
    revision_prompt = f"""Please revise your story based on this feedback. 
    Maintain the core concept but improve based on the suggestions.
    
    Original Story:
    {original_story}
    
    Critique:
    {critique}
    
    Please provide your revised story:"""
    
    response = user_proxy.initiate_chat(
        recipient=writer,
        message=revision_prompt,
        clear_history=True
    )
    
    # Get the actual revised story from the chat history
    revised_story = response.chat_history[1]["content"]  # Index 1 is the Writer's response
    logger.info(f"Writer completed revision ({len(revised_story)} characters)")
    
    return revised_story


def run_writing_workshop(agents: Dict[str, AssistantAgent], 
                        user_proxy: UserProxyAgent,
                        writing_prompt: str,
                        include_revision: bool = True) -> StoryResult:
    """
    Run the complete writing workshop workflow.
    
    Args:
        agents: Dictionary of agents
        user_proxy: User proxy agent
        writing_prompt: The initial writing prompt
        include_revision: Whether to include a revision round
        
    Returns:
        StoryResult with all outputs
    """
    try:
        # Get agents
        writer = agents["Writer"]
        critic = agents["Critic"]
        
        # Step 1: Write initial story
        logger.info("=" * 50)
        logger.info("STEP 1: Initial Story Writing")
        logger.info("=" * 50)
        original_story = write_initial_story(user_proxy, writer, writing_prompt)
        
        # Step 2: Get critique
        logger.info("=" * 50)
        logger.info("STEP 2: Critical Review")
        logger.info("=" * 50)
        critique = get_critique(user_proxy, critic, original_story)
        
        # Step 3: Revise story (optional)
        revised_story = None
        if include_revision:
            logger.info("=" * 50)
            logger.info("STEP 3: Story Revision")
            logger.info("=" * 50)
            revised_story = revise_story(user_proxy, writer, original_story, critique)
        
        # Create result
        result = StoryResult(
            original_story=original_story,
            critique=critique,
            revised_story=revised_story
        )
        
        logger.info("Writing workshop completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in writing workshop: {str(e)}", exc_info=True)
        raise


# ========================================================================
#   Output Formatting
# ========================================================================

def format_workshop_output(result: StoryResult) -> str:
    """
    Format the workshop results for display.
    
    Args:
        result: StoryResult object
        
    Returns:
        Formatted string output
    """
    output = []
    output.append("\n" + "=" * 70)
    output.append("WRITING WORKSHOP RESULTS")
    output.append("=" * 70)
    
    output.append("\n📝 ORIGINAL STORY:")
    output.append("-" * 50)
    output.append(result.original_story)
    
    output.append("\n\n🔍 CRITIQUE:")
    output.append("-" * 50)
    output.append(result.critique)
    
    if result.revised_story:
        output.append("\n\n✨ REVISED STORY:")
        output.append("-" * 50)
        output.append(result.revised_story)
    
    output.append("\n" + "=" * 70 + "\n")
    
    return "\n".join(output)


# ========================================================================
#   Main Application
# ========================================================================

def main() -> None:
    """
    Main function that orchestrates the multi-agent writing workshop.
    """
    global logger
    
    try:
        # Initialize logging
        logger = setup_logging(session_name="writing_workshop")
        
        # Load API keys
        api_keys = load_api_keys()
        
        # Define agent configurations
        agent_configs = [
            AgentConfig(
                name="Writer",
                system_message=WRITER_MESSAGE,
                model_config=DEFAULT_WRITER_MODEL,
                max_consecutive_auto_reply=1
            ),
            AgentConfig(
                name="Critic",
                system_message=CRITIC_MESSAGE,
                model_config=DEFAULT_CRITIC_MODEL,  # Using Anthropic!
                max_consecutive_auto_reply=1
            )
        ]
        
        # Create agents
        agents = create_writing_agents(agent_configs, api_keys)
        
        # Create user proxy
        user_proxy = create_user_proxy(max_consecutive_auto_reply=5)
        
        # Run the writing workshop
        result = run_writing_workshop(
            agents=agents,
            user_proxy=user_proxy,
            writing_prompt=DEFAULT_WRITING_PROMPT,  # Using the global variable
            include_revision=True                   # Set to False to skip revision
        )
        
        # Display results
        formatted_output = format_workshop_output(result)
        print(formatted_output)
        
        # Save results to file
        output_file = Path(LOG_DIR) / f"workshop_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        logger.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("Please check the logs for more details.")
        sys.exit(1)
        
    finally:
        logger.info("=" * 70)
        logger.info("Multi-Agent Writing Session Ended")
        logger.info("=" * 70)


# ========================================================================
#   Entry Point
# ========================================================================

if __name__ == "__main__":
    main()