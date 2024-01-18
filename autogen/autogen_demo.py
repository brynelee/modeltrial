from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample

# Please modify the content, remove these four lines of comment and rename this file to OAI_CONFIG_LIST to run the sample code.
# If using pyautogen v0.1.x with Azure OpenAI, please replace "base_url" with "api_base" (line 13 and line 20 below). Use "pip list" to check version of pyautogen installed.
#
# NOTE: This configuration lists GPT-4 as the default model, as this represents our current recommendation, and is known to work well with AutoGen. If you use a model other than GPT-4, you may need to revise various system prompts (especially if using weaker models like GPT-3.5-turbo). Moreover, if you use models other than those hosted by OpenAI or Azure, you may incur additional risks related to alignment and safety. Proceed with caution if updating this default.
""" [
    {
        "model": "gpt-4",
        "api_key": "<your OpenAI API key here>"
    },
    {
        "model": "<your Azure OpenAI deployment name>",
        "api_key": "<your Azure OpenAI API key here>",
        "base_url": "<your Azure OpenAI API base here>",
        "api_type": "azure",
        "api_version": "2023-07-01-preview"
    },
    {
        "model": "<your Azure OpenAI deployment name>",
        "api_key": "<your Azure OpenAI API key here>",
        "base_url": "<your Azure OpenAI API base here>",
        "api_type": "azure",
        "api_version": "2023-07-01-preview"
    }
]
 """

# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

# ollama serve
# ollama list
# litellm --model ollama/llama2
config_list = [
    {
        "base_url": "http://127.0.0.1:8000",
        "api_key": "NULL",
        "model": "ollama/codellama:13b",
    }
]

llm_config={
    "timeout": 600,
    "config_list": config_list
}

""" assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})
user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.") """

# You can also set config_list directly as a list, for example, config_list = [{'model': 'gpt-4', 'api_key': '<your OpenAI API key here>'},]
assistant = AssistantAgent(name = "assistant", llm_config=llm_config)
user_proxy = UserProxyAgent(
    name = "user_proxy", 
    human_input_mode = "NEVER",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web"},
    system_message="""Reply TERMINATE if the task has been solved with full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet."""
) 
task="""

Plot a chart of NVDA and TESLA stock price for 2023. Save the result to a file named nvda_tesla.png

"""


user_proxy.initiate_chat(assistant, message=task)
# This initiates an automated chat between the two agents to solve the task


# draw me a picture of a handsome boy.
# Plot a chart of NVDA and TESLA stock price for 2023. Save the result to a file named nvda_tesla.png
# Write a python script to plot a sine wave and save it to disc as a png file sine_wave.png
# List out the top 5 rivers in africa and their length and return that as a markdown table. Do not try to write any code, just write the table
# Write a python function to calculate the square root of a number, and call it with the number 4.