# Abstract 

Real-world enterprise systems are highly complex, with intertwined databases, constraints, and workflows that create an intractably large state space. While frontier large language models (LLMs) are increasingly deployed as autonomous agents, they struggle with long-horizon enterprise tasks due to a lack of internal system understanding. To enable reliable reasoning and planning, agents must learn the underlying dynamics of enterprise systems - motivating the need for enterprise world models.
Existing simulation environments, largely derived from video or text-based games, fail to capture the scale, relational structure, and workflow dependencies characteristic of real enterprise systems. To address this gap, we introduce a realistic, interactive environment, built upon a ServiceNow-based mock enterprise system, called World of Workflows (WoW).

WoW serves as a challenging, dynamic, and heterogeneous environment with over 6,000 interlinked tables and 93 workflows. 
WoW abstracts the massive enterprise system into a tractable, partially observable environment using table audits that represent database state changes.
We also enable agents to explore WoW with API-based tools and collect trajectories.
To understand the limitations of frontier LLMs as enterprise world models, we benchmark four models with three evaluation tasks: state prediction, action prediction, and constraint-based task completion. We demonstrate that existing LLMs have the potential to solve high-precision tasks using the proposed table audits as states, but they fail to accurately predict these audits during rollouts. The results highly motivate the significance of WoW, which establishes a new paradigm: an enterprise-level interactive environment bridging the gap between current world model research and practical use cases in enterprise systems.

# Installation

1. Created Virtual Environment with UV/venv
```bash
uv venv --python 3.12
```

2. Install servicenow_mcp from Submodules
```bash
uv pip install -e submodules/servicenow-mcp/
```

3. Install the source package 
```
uv pip install -e . 
wow-install
``` 

3. Run evaluation script
```bash
.venv/bin/python -m src.unified_evaluation --help
```

