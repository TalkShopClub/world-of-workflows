""" 
This script is used to setup the ServiceNow instance for the World of Workflows project. 
It will: 
    - Assign the flow to catalog items referenced in MCP tools in WoW. 
"""

from src.instance import SNowInstance
import requests 

SNOW_API_HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}

def assign_flow_to_catalog_items(instance: SNowInstance):
    """
    Assign the flow to catalog items referenced in MCP tools in WoW.
    """
    catalog_items = [
        "Developer Laptop (Mac)",
        "iPad mini",
        "iPad pro",
        "Sales Laptop",
        "Standard Laptop"
    ]

    # Get sys ids of catalog items
    resp = requests.get(
        f"{instance.snow_url}/api/now/table/sc_cat_item",
        auth=instance.snow_credentials,
        headers=SNOW_API_HEADERS,
        params={
            "sysparm_query": f"nameIN{','.join(catalog_items)}",
            "sysparm_fields": "sys_id,name"
        }
    )
    resp.raise_for_status()
    catalog_item_sys_ids = [x['sys_id'] for x in resp.json()['result']]

    # Get sys id of flow to assign to catalog items
    resp = requests.get(
        f"{instance.snow_url}/api/now/table/sys_hub_flow",
        auth=instance.snow_credentials,
        headers=SNOW_API_HEADERS,
        params={
            "sysparm_query": "name=Service Catalog item request",
            "sysparm_fields": "sys_id,name"
        }
    )
    resp.raise_for_status()
    flow_sys_id = resp.json()['result'][0]['sys_id']

    # Assign flow to catalog items and remove workflow
    for catalog_item_sys_id in catalog_item_sys_ids:
        resp = requests.post(
            f"{instance.snow_url}/api/now/table/sc_cat_item/{catalog_item_sys_id}",
            auth=instance.snow_credentials,
            headers=SNOW_API_HEADERS,
            json={"flow_designer_flow": flow_sys_id, "workflow": ""}
        )
        resp.raise_for_status()