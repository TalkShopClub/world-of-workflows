from importlib import resources
from . import data_files

# ServiceNow configuration
SNOW_BROWSER_TIMEOUT = 30000  # Milliseconds

# Custom workflows that are included with the benchmark
_workflow_path = str(
    resources.files(data_files).joinpath(
        "setup_files/world_of_workflows.xml"
    )
)

WORKFLOWS = {
    "setup_env_workflows": {
        "name": "EWM-World",
        "update_set": _workflow_path,
    }
}


# Custom UI Themes
_themes_path = str(
    resources.files(data_files).joinpath("setup_files/themes.xml")
)

UI_THEMES_UPDATE_SET = {
    "name": "WorkArena UI Themes",
    "update_set": _themes_path,
    "variants": [
        "Astranova",
        "Charlies",
        "Great pasta",
        "Mighty capital",
        "Speedy tires",
        "Skyward",
        "Turbobots",
        "Ultrashoes",
        "Vitasphere",
        "Workarena",
    ],
}
