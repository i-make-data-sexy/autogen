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

# UPDATE: Default model configurations
DEFAULT_ANALYST_MODEL = MODEL_CONFIGS["openai"]["gpt-4"]                # Using GPT4 for analyst tasks
DEFAULT_REVIEWER_MODEL = MODEL_CONFIGS["mistral"]["mistral-medium"]     # Using Mistral for compliance review!
DEFAULT_STRATEGIST_MODEL = MODEL_CONFIGS["anthropic"]["sonnet"]         # Using Anthropic for strategy
LOG_DIR = "utility_reports"

# UPDATE: System messages for utility company agents
ANALYST_MESSAGE = """You are a senior data analyst for a utility company, specializing in 
utility operations, energy consumption patterns, and infrastructure analysis. 
You excel at identifying trends, anomalies, and opportunities for improvement 
in electric and gas distribution systems. Your reports are clear, data-driven, 
and focused on operational efficiency and customer service improvement."""

REVIEWER_MESSAGE = """You are a regulatory compliance and risk management expert 
for a . You review reports for accuracy, regulatory compliance, safety 
considerations, and potential risks. You ensure all recommendations align with 
utility industry standards, environmental regulations, and company policies. 
You provide constructive feedback to strengthen reports before executive review."""

STRATEGIST_MESSAGE = """You are a strategic planning executive for a , 
focused on long-term sustainability, grid modernization, and customer satisfaction. 
You synthesize technical reports into actionable strategic recommendations, 
considering factors like renewable energy integration, infrastructure investment, 
and community impact. You excel at translating operational insights into 
executive-level strategic initiatives."""

# UPDATE: Default analysis prompt for utility operations
DEFAULT_ANALYSIS_PROMPT = """
Analyze the following utility operations scenario and provide recommendations:

SCENARIO: Summer Peak Load Management
- Peak electricity demand has increased 15% year-over-year
- Several neighborhoods experiencing frequent voltage fluctuations
- Customer complaints about power quality have increased 25%
- Current infrastructure is at 92% capacity during peak hours
- Budget constraints limit immediate infrastructure upgrades

Please provide:
1. Root cause analysis
2. Short-term mitigation strategies
3. Long-term infrastructure recommendations
4. Customer communication plan
5. Cost-benefit analysis of proposed solutions

"""


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
class AnalysisResult:
    """Container for utility analysis results."""
    initial_analysis: str
    compliance_review: str
    strategic_recommendations: Optional[str] = None
    full_history: List[Dict[str, Any]] = None


# ========================================================================
#   Logging Setup
# ========================================================================

def setup_logging(log_dir: str = LOG_DIR, session_name: str = "utility_analysis") -> logging.Logger:
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
    logger.info(f"Multi-Agent Analysis Session Started")
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
    # Set api_key variable
    api_key = api_keys.get(provider)
    
    # If key is missing
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
    # Debug logging
    logger.info(f"Creating config for provider: {model_config['provider']}")
    logger.info(f"Using API key starting with: {api_key[:8]}...")
    
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
    elif model_config["provider"] == "mistral":
        llm_config = {
            "config_list": [
                {
                    "model": model_config["model"],
                    "api_key": api_key,
                    "api_type": "mistral",  # Added api_type for Mistral
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

def create_utility_agents(agent_configs: List[AgentConfig], 
                         api_keys: Dict[str, Optional[str]]) -> Dict[str, AssistantAgent]:
    """
    Create all utility analysis agents based on configurations.
    
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
        human_input_mode="NEVER",                                   # Fully autonomous - never asks human
        max_consecutive_auto_reply=max_consecutive_auto_reply,
        code_execution_config=False,
    )
    logger.info("Created user proxy agent")
    return user_proxy


# ========================================================================
#   Utility Analysis Workflow
# ========================================================================

def perform_initial_analysis(user_proxy: UserProxyAgent, 
                           analyst: AssistantAgent, 
                           prompt: str) -> str:
    """
    Have the analyst perform initial analysis of the utility scenario.
    
    Args:
        user_proxy: User proxy agent
        analyst: Analyst agent
        prompt: Analysis prompt
        
    Returns:
        The analysis report
    """
    logger.info(f"Starting utility analysis")
    
    response = user_proxy.initiate_chat(
        recipient=analyst,
        message=prompt,
        clear_history=True
    )
    
    # Get the actual analysis from the chat history
    analysis = response.chat_history[1]["content"]  # Index 1 is the Analyst's response
    logger.info(f"Analyst completed analysis ({len(analysis)} characters)")
    
    return analysis


def review_for_compliance(user_proxy: UserProxyAgent, 
                         reviewer: AssistantAgent, 
                         analysis: str) -> str:
    """
    Have the reviewer check the analysis for compliance and risks.
    
    Args:
        user_proxy: User proxy agent
        reviewer: Reviewer agent
        analysis: The analysis to review
        
    Returns:
        The compliance review
    """
    logger.info("Getting compliance and risk review")
    
    review_prompt = f"""Please review this utility operations analysis for:
    1. Regulatory compliance issues
    2. Safety considerations
    3. Environmental impact
    4. Risk assessment
    5. Data accuracy and completeness
    
    Provide specific feedback and flag any concerns.
    
    Analysis to review:
    {analysis}"""
    
    response = user_proxy.initiate_chat(
        recipient=reviewer,
        message=review_prompt,
        clear_history=True
    )
    
    # Get the actual review from the chat history
    review = response.chat_history[1]["content"]  # Index 1 is the Reviewer's response
    logger.info(f"Reviewer completed compliance check ({len(review)} characters)")
    
    return review


def develop_strategy(user_proxy: UserProxyAgent, 
                    strategist: AssistantAgent, 
                    analysis: str, 
                    review: str) -> str:
    """
    Have the strategist develop executive recommendations.
    
    Args:
        user_proxy: User proxy agent
        strategist: Strategist agent
        analysis: The original analysis
        review: The compliance review
        
    Returns:
        Strategic recommendations
    """
    logger.info("Developing strategic recommendations")
    
    strategy_prompt = f"""Based on the analysis and compliance review below, 
    develop executive-level strategic recommendations that address:
    1. Immediate actions (0-3 months)
    2. Medium-term initiatives (3-12 months)
    3. Long-term strategic vision (1-5 years)
    4. Budget and resource allocation
    5. Stakeholder communication plan
    
    Original Analysis:
    {analysis}
    
    Compliance Review:
    {review}
    
    Provide clear, actionable recommendations suitable for executive presentation."""
    
    response = user_proxy.initiate_chat(
        recipient=strategist,
        message=strategy_prompt,
        clear_history=True
    )
    
    # Get the actual strategy from the chat history
    strategy = response.chat_history[1]["content"]  # Index 1 is the Strategist's response
    logger.info(f"Strategist completed recommendations ({len(strategy)} characters)")
    
    return strategy


def run_utility_analysis(agents: Dict[str, AssistantAgent], 
                        user_proxy: UserProxyAgent,
                        analysis_prompt: str,
                        include_strategy: bool = True) -> AnalysisResult:
    """
    Run the complete utility analysis workflow.
    
    Args:
        agents: Dictionary of agents
        user_proxy: User proxy agent
        analysis_prompt: The initial analysis prompt
        include_strategy: Whether to include strategic recommendations
        
    Returns:
        AnalysisResult with all outputs
    """
    try:
        # Get agents
        analyst = agents["Analyst"]
        reviewer = agents["Reviewer"]
        strategist = agents.get("Strategist")
        
        # Step 1: Initial analysis
        logger.info("=" * 50)
        logger.info("STEP 1: Initial Utility Analysis")
        logger.info("=" * 50)
        initial_analysis = perform_initial_analysis(user_proxy, analyst, analysis_prompt)
        
        # Step 2: Compliance review
        logger.info("=" * 50)
        logger.info("STEP 2: Compliance and Risk Review")
        logger.info("=" * 50)
        compliance_review = review_for_compliance(user_proxy, reviewer, initial_analysis)
        
        # Step 3: Strategic recommendations (optional)
        strategic_recommendations = None
        if include_strategy and strategist:
            logger.info("=" * 50)
            logger.info("STEP 3: Strategic Recommendations")
            logger.info("=" * 50)
            strategic_recommendations = develop_strategy(
                user_proxy, strategist, initial_analysis, compliance_review
            )
        
        # Create result
        result = AnalysisResult(
            initial_analysis=initial_analysis,
            compliance_review=compliance_review,
            strategic_recommendations=strategic_recommendations
        )
        
        logger.info("Utility analysis workflow completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in utility analysis: {str(e)}", exc_info=True)
        raise


# ========================================================================
#   Output Formatting
# ========================================================================

def format_analysis_output(result: AnalysisResult) -> str:
    """
    Format the analysis results for display.
    
    Args:
        result: AnalysisResult object
        
    Returns:
        Formatted string output
    """
    output = []
    output.append("UTILITY ANALYSIS REPORT")
    output.append("\nINITIAL ANALYSIS:")
    output.append(result.initial_analysis)
    output.append("\n\nCOMPLIANCE & RISK REVIEW:")
    output.append(result.compliance_review)
    
    if result.strategic_recommendations:
        output.append("\n\nSTRATEGIC RECOMMENDATIONS:")
        
        output.append(result.strategic_recommendations)
    
    output.append("\n" + "=" * 70 + "\n")
    
    return "\n".join(output)


# ========================================================================
#   Main Application
# ========================================================================

def main() -> None:
    """
    Main function that orchestrates the multi-agent utility analysis.
    """
    global logger
    
    try:
        # Initialize logging
        logger = setup_logging(session_name="utility_analysis")
        
        # Load API keys
        api_keys = load_api_keys()
        
        # UPDATE: Define agent configurations
        agent_configs = [
            AgentConfig(
                name="Analyst",
                system_message=ANALYST_MESSAGE,
                model_config=DEFAULT_ANALYST_MODEL,      
                max_consecutive_auto_reply=1
            ),
            AgentConfig(
                name="Reviewer",
                system_message=REVIEWER_MESSAGE,
                model_config=DEFAULT_REVIEWER_MODEL,         
                max_consecutive_auto_reply=1
            ),
            AgentConfig(
                name="Strategist",
                system_message=STRATEGIST_MESSAGE,
                model_config=DEFAULT_STRATEGIST_MODEL,         
                max_consecutive_auto_reply=1
            )
        ]
        
        # Create agents
        agents = create_utility_agents(agent_configs, api_keys)
        
        # Create user proxy
        user_proxy = create_user_proxy(max_consecutive_auto_reply=5)
        
        # Run the analysis workflow
        result = run_utility_analysis(
            agents=agents,
            user_proxy=user_proxy,
            analysis_prompt=DEFAULT_ANALYSIS_PROMPT,  # Using the global variable
            include_strategy=True  # UPDATE: Set to False to skip strategic planning
        )
        
        # Display results
        formatted_output = format_analysis_output(result)
        print(formatted_output)
        
        # Save results to file
        output_file = Path(LOG_DIR) / f"analysis_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
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
        logger.info("Utilities Company Multi-Agent Analysis Session Ended")
        logger.info("=" * 70)


# ========================================================================
#   Entry Point
# ========================================================================

if __name__ == "__main__":
    main()