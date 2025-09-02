# AutoGen Multi-Agent Orchestration System

A template for building multi-agent AI systems using Microsoft's AutoGen framework. This project demonstrates how to coordinate multiple AI models from different providers to solve complex analytical tasks.

## Overview

This system showcases two practical implementations:
- **Creative Writing Workshop**: Multiple agents collaborate to write, critique, and revise stories
- **Utility Infrastructure Analysis**: Agents analyze operational challenges, ensure compliance, and develop strategic recommendations

## Features

- Multi-provider support (OpenAI, Anthropic, Mistral, Google Gemini, Meta)
- Clean separation of concerns with specialized agents
- Comprehensive logging system
- Modular, extensible architecture
- Type-safe configuration with dataclasses
- Automated workflow orchestration

## Project Structure

```
├── analyst_reviewer_strategist.py  # Utility analysis implementation
├── utility_reports/            # Analysis output directory
├── .env                        # API keys (create this for your project)
└── requirements.txt            # Python dependencies
├── README.md                   # Overview of repo
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root with your API keys:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
META_API_KEY=...
```

## Usage

```bash
python analyst_reviewer_strategist.py
```

This will:
- Analyze the infrastructure scenario
- Review for compliance and risks
- Develop strategic recommendations
- Generate a comprehensive report

## Customization

### Changing Models

Edit the default model configurations at the top of each file:

```python
# UPDATE: Default model configurations
DEFAULT_WRITER_MODEL = MODEL_CONFIGS["openai"]["gpt-4"]
DEFAULT_CRITIC_MODEL = MODEL_CONFIGS["anthropic"]["sonnet"]
```

### Modifying Prompts

Update the system messages and default prompts:

```python
# UPDATE: System messages for agents
WRITER_MESSAGE = """Your custom writer instructions..."""

# UPDATE: Default writing prompt
DEFAULT_WRITING_PROMPT = """Your custom prompt..."""
```

### Adding New Agents

1. Define the agent configuration
2. Add system message constant
3. Create workflow function
4. Insert into main workflow

## Code Organization

- **Imports**: Standard library, third-party, and local imports
- **Constants**: Model configurations and system messages
- **Data Classes**: Type-safe structures for configurations and results
- **Logging**: Comprehensive logging setup
- **API Management**: Secure key loading and validation
- **Agent Creation**: Factory functions for building agents
- **Workflow**: Discrete functions for each analysis step
- **Main**: Orchestration and entry point

## API Key Requirements

- **OpenAI**: Get from https://platform.openai.com/account/api-keys
- **Anthropic**: Get from https://console.anthropic.com/
- **Google Gemini**: Get from https://makersuite.google.com/app/apikey
- **Mistral**: Get from https://console.mistral.ai/
- **Meta**: Contact Meta for API access

## Troubleshooting

### Invalid API Key Error
- Verify your `.env` file exists and contains valid keys
- Check that API keys start with the correct prefix (e.g., `sk-` for OpenAI)
- Ensure no extra spaces or quotes around keys

### Model Not Found
- Some model names may change over time
- Check provider documentation for current model names
- Update MODEL_CONFIGS dictionary as needed

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+ recommended)

## Example Output

The system generates detailed reports with three perspectives:

1. **Initial Analysis**: Technical assessment of the problem
2. **Compliance Review**: Regulatory and risk evaluation  
3. **Strategic Recommendations**: Phased action plans with budgets

## Contributing

This project serves as a template. Feel free to:
- Add new agent types
- Implement different workflows
- Support additional providers
- Enhance output formatting

## License

[Your license choice]

## Acknowledgments

Built with Microsoft's AutoGen framework
- Documentation: https://microsoft.github.io/autogen/
- GitHub: https://github.com/microsoft/autogen