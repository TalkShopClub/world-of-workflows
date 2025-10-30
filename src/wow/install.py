import html
import json
import logging
from os import wait
import re
import tenacity

from datetime import datetime
from playwright.sync_api import sync_playwright
from tenacity import retry, stop_after_attempt, retry_if_exception_type
from time import sleep

from .api.ui_themes import get_workarena_theme_variants
from .api.utils import table_api_call
from .config import (
    # For workflows setup
    WORKFLOWS,
    # For UI themes setup
    UI_THEMES_UPDATE_SET,
)
from .api.user import set_user_preference
from .instance import SNowInstance
from .utils import url_login
from .environment.states import setup_auditing


def _set_sys_property(property_name: str, value: str):
    """
    Set a sys_property in the instance.

    """
    instance = SNowInstance()

    property = table_api_call(
        instance=instance,
        table="sys_properties",
        params={"sysparm_query": f"name={property_name}", "sysparm_fields": "sys_id"},
    )["result"]

    if not property:
        property_sysid = ""
        method = "POST"
    else:
        property_sysid = "/" + property[0]["sys_id"]
        method = "PUT"

    if method == "PUT": 
        property = table_api_call(
            instance=instance,
            table="sys_properties",
            method="PUT",
            json={"name": property_name, "value": value},
            put_filter={"sys_id": property_sysid.strip("/")},
        )

    else: 
        property = table_api_call(
            instance=instance,
            table=f"sys_properties{property_sysid}",
            method=method,
            json={"name": property_name, "value": value},
        )

    # Verify that the property was updated
    assert property["result"]["value"] == value, f"Error setting {property_name}."


def _get_sys_property(property_name: str) -> str:
    """
    Get a sys_property from the instance.

    """
    instance = SNowInstance()

    property_value = table_api_call(
        instance=instance,
        table="sys_properties",
        params={"sysparm_query": f"name={property_name}", "sysparm_fields": "value"},
    )["result"][0]["value"]

    return property_value


def _install_update_set(path: str, name: str):
    """
    Install a ServiceNow update set

    Parameters:
    -----------
    path: str
        The path to the update set file.
    name: str
        The name of the update set as it should appear in the UI.

    Notes: requires interacting with the UI, so we use playwright instead of the API

    """
    with sync_playwright() as playwright:
        instance = SNowInstance()
        browser = playwright.chromium.launch(headless=True, slow_mo=1000)
        page = browser.new_page()
        url_login(instance, page)

        # Navigate to the update set upload page and upload all update sets
        logging.info(f"Uploading update set {name}...")
        page.goto(
            instance.snow_url
            + "/now/nav/ui/classic/params/target/upload.do%3Fsysparm_referring_url%3Dsys_remote_update_set_list.do%253Fsysparm_fixed_query%253Dsys_class_name%253Dsys_remote_update_set%26sysparm_target%3Dsys_remote_update_set"
        )
        iframe = page.wait_for_selector('iframe[name="gsft_main"]').content_frame()
        with page.expect_file_chooser() as fc_info:
            iframe.locator("#attachFile").click()
        file_chooser = fc_info.value
        file_chooser.set_files(path)
        iframe.locator("input:text('Upload')").click()
        sleep(5)

        # Apply all update sets
        logging.info(f"Applying update set {name}...")
        # ... retrieve all update sets that are ready to be applied
        update_set = table_api_call(
            instance=instance,
            table="sys_remote_update_set",
            params={
                "sysparm_query": f"name={name}^state=loaded",
            },
        )["result"][0]
        # ... apply them
        logging.info(f"... {update_set['name']}")
        page.goto(instance.snow_url + "/sys_remote_update_set.do?sys_id=" + update_set["sys_id"])
        page.locator("button:has-text('Preview Update Set')").first.click()
        page.wait_for_selector("text=Succeeded", timeout=60000)
        # click escape to close popup
        page.keyboard.press("Escape")
        page.locator("button:has-text('Commit Update Set')").first.click()
        
        # Check for "Confirm Data Loss" popup and handle if it appears
        try:
            # Wait briefly to see if the popup appears
            page.wait_for_selector("text=Confirm Data Loss", timeout=3000)
            logging.info("Confirm Data Loss popup detected, clicking 'Proceed with Commit'...")
            page.locator("button:has-text('Proceed with Commit')").first.click()
        except:
            # Popup didn't appear, continue normally
            logging.info("No Confirm Data Loss popup detected, proceeding...")
        
        page.wait_for_selector("text=Succeeded", timeout=60000)

        browser.close()


def setup_workflows():
    """
    Verify that workflows are correctly installed.
    If not, install them.

    """
    if not check_workflows_installed():
        install_workflows()
        assert check_workflows_installed(wait_for_completion=True), "Workflow installation failed."
        logging.info("Workflow installation succeeded.")


def check_workflows_installed(wait_for_completion=False):
    """
    Check if the workflows are installed in the instance.

    Will return False if workflows need to be (re)installed. True if all is good.

    """
    expected_workflow_names = ["Committing update set: " + x["name"] for x in WORKFLOWS.values()]
    import time
    
    max_attempts = 15  # 12 attempts * 5 seconds = 60 seconds max wait
    attempt = 0
    
    while attempt < max_attempts:
        workflows = table_api_call(
            instance=SNowInstance(),
            table="sys_progress_worker",
            params={
                "sysparm_query": "nameIN" + ",".join(expected_workflow_names),
            },
        )["result"]

        workflows = [w for w in workflows if w["state"] == "complete"]

        # Verify that all workflows are installed
        if len(workflows) != len(WORKFLOWS):
            if attempt < max_attempts - 1 and wait_for_completion:  # Don't log on the last attempt
                logging.info(
                    f"Workflow has not been fully committed yet (attempt {attempt + 1}/{max_attempts}): {set(expected_workflow_names) - set([w['name'] for w in workflows])}. Rechecking in 5 seconds..."
                )
                time.sleep(10)
                attempt += 1
            else:
                if wait_for_completion:
                    logging.info(
                        f"Missing workflows after {max_attempts} attempts: {set(expected_workflow_names) - set([w['name'] for w in workflows])}."
                    )
                return False
        else:
            break

    logging.info("All workflows are installed properly.")
    return True


def install_workflows():
    """
    Install workflows using ServiceNow update sets.

    Notes: requires interacting with the UI, so we use playwright instead of the API

    """
    logging.info("Installing workflow update sets...")
    for wf in WORKFLOWS.values():
        _install_update_set(path=wf["update_set"], name=wf["name"])


def enable_url_login():
    """
    Configure the instance to allow login via URL.

    """
    _set_sys_property(property_name="glide.security.restrict.get.login", value="false")
    logging.info("URL login enabled.")


def disable_password_policies():
    """
    Disable password policies in the instance.

    Notes: this is required to allow the creation of users with weak passwords.

    """
    _set_sys_property(property_name="glide.security.password.policy.enabled", value="false")
    logging.info("Password policies disabled.")


def disable_guided_tours():
    """
    Hide guided tour popups

    """
    _set_sys_property(property_name="com.snc.guided_tours.sp.enable", value="false")
    _set_sys_property(property_name="com.snc.guided_tours.standard_ui.enable", value="false")
    logging.info("Guided tours disabled.")


def disable_welcome_help_popup():
    """
    Disable the welcome help popup

    """
    set_user_preference(instance=SNowInstance(), key="overview_help.visited.navui", value="true")
    logging.info("Welcome help popup disabled.")


def disable_analytics_popups():
    """
    Disable analytics popups (needs to be done through UI since Vancouver release)

    """
    _set_sys_property(property_name="glide.analytics.enabled", value="false")
    logging.info("Analytics popups disabled.")


def setup_ui_themes():
    """
    Install custom UI themes and set it as default

    """
    logging.info("Installing custom UI themes...")
    _install_update_set(path=UI_THEMES_UPDATE_SET["update_set"], name=UI_THEMES_UPDATE_SET["name"])
    check_ui_themes_installed()

    logging.info("Setting default UI theme")
    _set_sys_property(
        property_name="glide.ui.polaris.theme.custom",
        value=get_workarena_theme_variants(SNowInstance())[0]["theme.sys_id"],
    )

    # Set admin user's theme variant
    # ... get user's sysid
    admin_user = table_api_call(
        instance=SNowInstance(),
        table="sys_user",
        params={"sysparm_query": "user_name=admin", "sysparm_fields": "sys_id"},
    )["result"][0]
    # ... set user preference
    set_user_preference(
        instance=SNowInstance(),
        user=admin_user["sys_id"],
        key="glide.ui.polaris.theme.variant",
        value=[
            x["style.sys_id"]
            for x in get_workarena_theme_variants(SNowInstance())
            if x["style.name"] == "Workarena"
        ][0],
    )


def check_ui_themes_installed():
    """
    Check if the UI themes are installed in the instance.

    """
    expected_variants = set([v.lower() for v in UI_THEMES_UPDATE_SET["variants"]])
    installed_themes = get_workarena_theme_variants(SNowInstance())
    installed_themes = set([t["style.name"].lower() for t in installed_themes])

    assert (
        installed_themes == expected_variants
    ), f"""UI theme installation failed.
        Expected: {expected_variants}
        Installed: {installed_themes}
        """


def set_home_page():
    logging.info("Setting default home page")
    _set_sys_property(property_name="glide.login.home", value="/now/nav/ui/home")


def wipe_system_admin_preferences():
    """
    Wipe all system admin preferences

    """
    logging.info("Wiping all system admin preferences")
    sys_admin_prefs = table_api_call(
        instance=SNowInstance(),
        table="sys_user_preference",
        params={"sysparm_query": "user.user_name=admin", "sysparm_fields": "sys_id,name"},
    )["result"]

    # Delete all sysadmin preferences
    logging.info("... Deleting all preferences")
    for pref in sys_admin_prefs:
        logging.info(f"...... deleting {pref['name']}")
        table_api_call(
            instance=SNowInstance(), table=f"sys_user_preference/{pref['sys_id']}", method="DELETE"
        )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    reraise=True,
    before_sleep=lambda _: logging.info("An error occurred. Retrying..."),
)
def setup():
    """
    Check that World of Workflows is installed correctly in the instance.

    """

    # Enable URL login (XXX: Do this first since other functions can use URL login)
    enable_url_login()

    # Disable password policies
    disable_password_policies()

    # Set default landing page
    set_home_page()

    # Disable popups for new users
    # ... guided tours
    disable_guided_tours()
    # ... analytics
    disable_analytics_popups()
    # ... help
    disable_welcome_help_popup()

    # Install custom UI themes (needs to be after disabling popups)
    setup_ui_themes()

    # Clear all predefined system admin preferences (e.g., default list views, etc.)
    wipe_system_admin_preferences()

    # XXX: Install flows
    setup_workflows()

    # Setup auditing in ServiceNow instance to allow for state tracking
    setup_auditing()

    # Save installation date
    logging.info("Saving installation date")
    _set_sys_property(property_name="world_of_workflows.installation.date", value=datetime.now().isoformat())

    logging.info("World of Workflows setup complete.")


def main():
    """
    Entrypoint for package CLI installation command

    """
    logging.basicConfig(level=logging.INFO)

    try:
        past_install_date = _get_sys_property("world_of_workflows.installation.date")
        logging.info(f"Detected previous installation on {past_install_date}. Reinstalling...")
    except:
        past_install_date = "never"

    logging.info(
        f"""

██     ██  ██████  ██████  ██      ██████      ██████  ███████
██     ██ ██    ██ ██   ██ ██      ██   ██    ██    ██ ██     
██  █  ██ ██    ██ ██████  ██      ██   ██    ██    ██ █████  
██ ███ ██ ██    ██ ██   ██ ██      ██   ██    ██    ██ ██     
 ███ ███   ██████  ██   ██ ███████ ██████      ██████  ██     

██     ██  ██████  ██████  ██   ██ ███████ ██       ██████  ██     ██ ███████
██     ██ ██    ██ ██   ██ ██  ██  ██      ██      ██    ██ ██     ██ ██     
██  █  ██ ██    ██ ██████  █████   █████   ██      ██    ██ ██  █  ██ ███████
██ ███ ██ ██    ██ ██   ██ ██  ██  ██      ██      ██    ██ ██ ███ ██      ██
 ███ ███   ██████  ██   ██ ██   ██ ██      ███████  ██████   ███ ███  ███████

Instance: {SNowInstance().snow_url}
Previous installation: {past_install_date}

"""
    )
    setup()

if __name__ == "__main__":
    main()