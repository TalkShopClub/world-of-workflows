from faker import Faker
import numpy as np
import time

fake = Faker()

from ..instance import SNowInstance
from .utils import table_api_call


def set_user_preference(instance: SNowInstance, key: str, value: str, user=None) -> dict:
    """
    Set a user preference in the ServiceNow instance

    Parameters:
    -----------
    key: str
        The name of the preference
    value: str
        The value of the preference
    user: str
        The sys_id of the user. If None, the preference will be set globally.

    Returns:
    --------
    dict
        The preference that was set

    """
    if user is None:
        # make it global
        user = ""
        system = True
    else:
        system = False

    # Try to get the preference's sys_id
    preference = table_api_call(
        instance=instance,
        table="sys_user_preference",
        params={"sysparm_query": f"name={key},user={user}", "sysparm_fields": "sys_id"},
    )["result"]

    if not preference:
        # ... The preference key doesn't exist, create it
        pref_sysid = ""
        method = "POST"
    else:
        # ... The preference key exists, update it
        pref_sysid = "/" + preference[0]["sys_id"]
        method = "PUT"

    property = table_api_call(
        instance=instance,
        table=f"sys_user_preference{pref_sysid}",
        method=method,
        json={
            "name": key,
            "value": value,
            "user": user,
            "system": system,
            "description": "Updated by World of Workflows",
        },
    )["result"]

    # Verify that the property was updated
    property["user"] = (
        property["user"].get("value") if isinstance(property["user"], dict) else property["user"]
    )
    assert (
        property["value"] == value
    ), f"Error setting system property {key}, incorrect value {property['value']}, while expecting {value}."
    assert (
        property["user"] == user
    ), f"Error setting system property {key}, incorrect user {property['user']}, while expecting {user}."
    assert (
        property["system"] == str(system).lower()
    ), f"Error setting {key}, incorrect system {property['system']}, while expecting {system}."

    return property
