# Abstract 

Real-world enterprise systems are highly complex, with intertwined databases, constraints, and workflows that create an intractably large state space. While frontier large language models (LLMs) are increasingly deployed as autonomous agents, they struggle with long-horizon enterprise tasks due to a lack of internal system understanding. To enable reliable reasoning and planning, agents must learn the underlying dynamics of enterprise systems - motivating the need for enterprise world models.
Existing simulation environments, largely derived from video or text-based games, fail to capture the scale, relational structure, and workflow dependencies characteristic of real enterprise systems. To address this gap, we introduce a realistic, interactive environment, built upon a ServiceNow-based mock enterprise system, called World of Workflows (WoW).

WoW serves as a challenging, dynamic, and heterogeneous environment with over 6,000 interlinked tables and 93 workflows. 
WoW abstracts the massive enterprise system into a tractable, partially observable environment using table audits that represent database state changes.
We also enable agents to explore WoW with API-based tools and collect trajectories.
To understand the limitations of frontier LLMs as enterprise world models, we benchmark four models with three evaluation tasks: state prediction, action prediction, and constraint-based task completion. We demonstrate that existing LLMs have the potential to solve high-precision tasks using the proposed table audits as states, but they fail to accurately predict these audits during rollouts. The results highly motivate the significance of WoW, which establishes a new paradigm: an enterprise-level interactive environment bridging the gap between current world model research and practical use cases in enterprise systems.

# Installation

1. Create a developer instance in ServiceNow 

    - Go to https://developer.servicenow.com/dev.do 
    - Press "Sign In" in the top right part of the page 
    - Press "New User? Get a ServiceNow ID" 
    - Fill the form and your request for an instance will be initiated (This can take upto 10 mins to complete) 

2. You should now see your URL and credentials. Based on this information, set the following environment variables:
    - `SNOW_INSTANCE_URL`: The URL of your ServiceNow developer instance
    - `SNOW_INSTANCE_UNAME`: The username, should be "admin"
    - `SNOW_INSTANCE_PWD`: The password, make sure you place the value in quotes "" and be mindful of [escaping special shell characters](https://onlinelinuxtools.com/escape-shell-characters). Running `echo $SNOW_INSTANCE_PWD` should print the correct password.

3. Log into your instance via a browser using the admin credentials. Close any popup that appears on the main screen (e.g., agreeing to analytics).

4. Set other environment variables: 
    - OPENROUTER_API_KEY (You can get one from [OpenRouter](https://openrouter.ai/))
    - LANGFUSE_PUBLIC_KEY (For these, create an account on [Langfuse](https://langfuse.com/) and create a project. Each project has these associated keys that will be provided)
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST  

5. Create Virtual Environment with uv
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

6. Install servicenow_mcp from Submodules
```bash
uv pip install -e submodules/servicenow-mcp/
```

7. Install the source package 
```
uv pip install -e . 
wow-install
``` 

8. Run example script
```
python example_usage.py
```

# Usage 


