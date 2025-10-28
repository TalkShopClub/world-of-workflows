import requests  
import os 
import re 
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import os 
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from rest_apis.world_model_scripts.world_model_agent import SysAuditRecord
from src.browsergym.workarena.instance import SNowInstance
from src.browsergym.workarena.api.utils import table_api_call 
from datetime import datetime, timedelta
  
def get_workarena_snow_config():
    """
    Get the workarena snow config. Returns the table base url, auth, and headers. 
    """
    table_base_url = f"{os.getenv('SNOW_INSTANCE_URL')}/api/now/table/" 
    auth = (os.getenv('SNOW_INSTANCE_UNAME'), os.getenv('SNOW_INSTANCE_PWD'))
    headers = {
        "Accept": "application/json", 
        "Content-Type": "application/json"
    } 

    return table_base_url, auth, headers 

def toggle_audit_inserts():
    """
    Turn on audit_inserts system property. This allows for auditing of insert records of auditing enabled tables. 
    """ 
    table_base_url, auth, headers = get_workarena_snow_config()

    # Get the sys_id of the audit_inserts system property
    resp = requests.get(
        table_base_url + "sys_properties", 
        auth = auth, 
        headers = headers, 
        params = {"sysparm_query": "name=glide.sys.audit_inserts", "sysparm_limit": 1}
    ) 

    resp.raise_for_status()
    audit_inserts_sys_id = resp.json().get('result')[0].get('sys_id')


    resp = requests.put(
        table_base_url + "sys_properties" + f"/{audit_inserts_sys_id}", 
        auth = auth, 
        headers = headers, 
        json = {"value": "true"}
    )

    resp.raise_for_status()
    return resp  

def get_all_tables(return_sys_tables=False): 
    """ 
    Get all tables from sys_db_object  
    """ 

    table_base_url, auth, headers = get_workarena_snow_config()
    resp = requests.get(
        table_base_url + "sys_db_object", 
        auth = auth, 
        headers = headers, 
        params = {"sysparm_query": "ORDERBYname", "sysparm_limit": 100000}
    )
    resp.raise_for_status()
    if return_sys_tables:
        return [table.get('name') for table in resp.json().get('result') if table.get('name') != 'sys_db_object']
    else: 
        # Dont return any tables that has "sys" in its name or starts with "sn_". 
        # exception: sys_user is allowed.
        return [table.get('name') for table in resp.json().get('result') 
        if table.get('name') != 'sys_db_object' and not table.get('name').startswith('sn_') 
        and not 'sys' in table.get('name')] + ['sys_user', 'sys_user_group', 'sys_user_grmember', 'sys_user_has_role', 'sys_user_role']

def get_table_schema(table_name: str) -> List[Dict]:
    """
    Fetch the schema (fields/columns) for a given table from ServiceNow.
    Returns a list of field metadata dictionaries.
    """
    table_base_url, auth, headers = get_workarena_snow_config()
    resp = requests.get(
        table_base_url + "sys_dictionary",
        auth=auth,
        headers=headers,
        params={
            "sysparm_query": f"name={table_name}",
            "sysparm_fields": "element,column_label,internal_type,max_length,mandatory,reference",
            "sysparm_limit": 10000
        }
    )
    resp.raise_for_status()
    return resp.json().get("result", [])

def toggle_auditing_for_table(table_name, toggle_to=True):
    """
    Toggle auditing for a table.
    """
    table_base_url, auth, headers = get_workarena_snow_config()

    # Get the sys_id of the table entry which applies to the entire table, not just a specific field 
    resp = requests.get(
        table_base_url + "sys_dictionary", 
        auth = auth, 
        headers = headers, 
        params = {"sysparm_query": f"name={table_name}^element ISEMPTY", "sysparm_limit": 1}
    )
    resp.raise_for_status()

    if len(resp.json().get('result')) == 0:
        return None 

    sys_id = resp.json().get('result')[0].get('sys_id') 

    resp = requests.put(
        table_base_url + "sys_dictionary" + f"/{sys_id}", 
        auth = auth, 
        headers = headers, 
        json = {"audit": "true" if toggle_to else "false"}
    ) 
    return resp 

def toggle_auditing_for_all_tables(toggle_to=True, return_sys_tables=False): 
    """
    Toggle auditing for all tables.
    """ 

    tables = get_all_tables(return_sys_tables)
    for table in tqdm(tables, desc="Toggling auditing for all tables"):
        resp = toggle_auditing_for_table(table, toggle_to)

    return None

def check_auditing_for_table(table_name):
    """
    Check if auditing is enabled for a table.
    """
    table_base_url, auth, headers = get_workarena_snow_config()
    resp = requests.get(
        table_base_url + "sys_dictionary", 
        auth = auth, 
        headers = headers, 
        params = {"sysparm_query": f"name={table_name}^element ISEMPTY", "sysparm_limit": 1}
    )
    resp.raise_for_status()
    return resp

def audits_to_states(audits, mcp_response):
    new_audits = []
    kept_field = ['fieldname', 'newvalue', 'documentkey', 'tablename', 'oldvalue']
    for audit in audits:
        new_audit = {}
        for field in kept_field:
            new_audit[field] = audit[field]
        new_audits.append(new_audit)
    additional_information = {
        'num_audits': len(new_audits),
        'num_modified_entries': [i for i in new_audits if i['newvalue'] != ['DELETE'] ],
        'num_deleted_entries': [i for i in new_audits if i['newvalue'] == ['DELETE'] ],
        'operation_type': 'get',
        'knowledge_extracted': False,
        'tables_modified': list(set([audit['tablename'] for audit in new_audits]))
    }
    return new_audits, additional_information

def setup_auditing(toggle_to=True, return_sys_tables=False): 
    """
    Setup auditing for all tables.
    """ 

    toggle_audit_inserts()
    toggle_auditing_for_all_tables(toggle_to, return_sys_tables)

def get_sys_audit(start_timestamp: str, end_timestamp: Optional[str] = None): 
    """  
    Get sys audit records from sys_audit table from last timestamp to now. 

    start_timestamp: str - last timestamp to get sys audit records from. Example: 2025-09-16 21:25:00
    end_timestamp: str - end timestamp to get sys audit records to. Example: 2025-09-16 21:30:00. If None, will get all records from last timestamp to now.
    """ 
    # Validate start timestamp
    if not isinstance(start_timestamp, str):
        raise ValueError("start_timestamp must be a string")
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', start_timestamp):
        raise ValueError("start_timestamp must be in the format YYYY-MM-DD HH:MM:SS")

    # Validate end timestamp
    if end_timestamp and not isinstance(end_timestamp, str):
        raise ValueError("end_timestamp must be a string")
    if end_timestamp and not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', end_timestamp):
        raise ValueError("end_timestamp must be in the format YYYY-MM-DD HH:MM:SS")

    # Prepare params based on whether end_timestamp is provided
    if end_timestamp:
        params = {"sysparm_query": f"sys_created_on>{start_timestamp}^sys_created_on<{end_timestamp}^ORDERBYDESCsys_created_on", "sysparm_limit": 100000}
    else:
        params = {"sysparm_query": f"sys_created_on>{start_timestamp}^ORDERBYDESCsys_created_on", "sysparm_limit": 100000}

    table_base_url, auth, headers = get_workarena_snow_config()
    resp = requests.get(
        table_base_url + "sys_audit", 
        auth = auth, 
        headers = headers, 
        params = params
    )

    resp.raise_for_status()
    return resp.json().get('result') 

def get_state_diff_timestamps(instance: SNowInstance, document_keys: Tuple[str, str], all_state_audit_records: List[Dict]): 
    """
    Get the timestamps for the first and last audit records using combination of documentkeys and first and last state audit records.

    Args: 
        document_keys: Tuple[str, str] - Tuple of document key for the first and last state audit record.
        all_state_audit_records: List[Dict] - List of all the state audit records between first and last audit records. Only these records will be used to construct the merged state.
    
    Returns: 
        Tuple[str, str] - Tuple of start time and end time for the state diff.
    """ 

    first_document_key, last_document_key = document_keys

    # Get document key for first and last audit record from sys_audit table 

    # Filter for first record
    first_filter = {
        "fieldname": all_state_audit_records[0]['fieldname'],
        "newvalue": all_state_audit_records[0]['newvalue'],
        "tablename": all_state_audit_records[0]['tablename'],
        "oldvalue": all_state_audit_records[0]['oldvalue'] 
    }
    query = f"documentkey={first_document_key}" 
    query += "^" + "^".join([f"{k}={v}" if v != "" else f"{k} ISEMPTY" for k, v in first_filter.items()])

    print(f"First query: {query}")

    # Get document key for first audit record from sys_audit table 
    resp = table_api_call(instance, table="sys_audit", params={"sysparm_query": query})
    # Ensure exactly one record is returned. This is important to ensure that only one unique record is returned for the first audit record. 
    assert len(resp['result']) == 1, f"Expected exactly one record for first audit record, but got {len(resp['result'])}"
    start_time = resp['result'][0]['sys_created_on']
    
    # Filter for last record 
    last_filter = {
        "fieldname": all_state_audit_records[-1]['fieldname'],
        "newvalue": all_state_audit_records[-1]['newvalue'],
        "tablename": all_state_audit_records[-1]['tablename'],
        "oldvalue": all_state_audit_records[-1]['oldvalue'] 
    }
    query = f"documentkey={last_document_key}" 
    query += "^" + "^".join([f"{k}={v}" for k, v in last_filter.items()]) 

    print(f"Last query: {query}")

    # Get document key for last audit record from sys_audit table 
    resp = table_api_call(instance, table="sys_audit", params={"sysparm_query": query})
    # Ensure exactly one record is returned. This is important to ensure that only one unique record is returned for the last audit record. 
    assert len(resp['result']) == 1, f"Expected exactly one record for last audit record, but got {len(resp['result'])}"
    end_time = resp['result'][0]['sys_created_on']

    # Add 20 seconds to the end time to ensure all state changes after last action are included 
    end_time = (datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=20)).strftime("%Y-%m-%d %H:%M:%S")

    return start_time, end_time 

def merge_state_diffs(instance: SNowInstance, document_keys: Tuple[str, str], first_and_last_state_audit_records: List[Dict], all_state_audit_records: List[Dict]): 
    """
    Merge all the state diffs between first and last audit records. Gathers document key for each audit record and then iteratively constructs the merged state. 

    Args: 
        instance: SNowInstance - ServiceNow instance to query
        document_keys: Tuple[str, str] - Tuple of document key for the first and last state audit record.
        first_and_last_state_audit_records: List[Dict] - List of first and last state audit record. These will be used to get the timestamps for the state diff.
        all_state_audit_records: List[Dict] - List of all the state audit records between first and last audit records. Only these records will be used to construct the merged state.

    Returns: 
        linked_records: List[Dict] - List of audit records from all_state_audit_records with their corresponding documentkey, sys_id, and sys_created_on fields linked from the ServiceNow sys_audit table.
    """ 

    # Get the timestamps for the first and last audit records 
    start_time, end_time = get_state_diff_timestamps(instance, document_keys, first_and_last_state_audit_records)

    # Get all the audit records between first and last audit records 
    resp = table_api_call(instance, table="sys_audit", params={"sysparm_query": f"sys_created_on>={start_time}^sys_created_on<={end_time}^ORDERBYsys_created_on"})

    # Link each audit record with corresponding documentkey using pointer-based approach
    # This ensures one-to-one mapping between target records and audit records
    audit_records = resp['result']
    used_audit_indices = set()  # Track which audit records have been used
    
    for i, target_record in enumerate(tqdm(all_state_audit_records, desc="Linking audit records with corresponding documentkey")):
        match_found = False
        
        # Search for matching audit record that hasn't been used yet
        for j, audit_record in enumerate(audit_records):
            # Skip if this audit record has already been matched
            if j in used_audit_indices:
                continue
                
            # Check if all required fields match
            if (audit_record.get('fieldname') == target_record.get('fieldname') and
                audit_record.get('newvalue') == target_record.get('newvalue') and
                audit_record.get('tablename') == target_record.get('tablename') and
                audit_record.get('oldvalue') == target_record.get('oldvalue')):
                
                # Link the documentkey to the target record
                target_record['documentkey'] = audit_record.get('documentkey')
                used_audit_indices.add(j)  # Mark this audit record as used
                match_found = True
                break  # Found a match, move to next target record
        
        # Raise error if no match was found for this target record
        if not match_found:
            raise ValueError(f"No matching audit record found for target_record {i}: {target_record}")

    # Iteratively construct the merged state
    merged_state = {}
    
    # Process each audit record to build the merged state
    for i, audit_record in tqdm(enumerate(all_state_audit_records), desc="Constructing merged state"):
        table_name = audit_record.get('tablename')
        document_key = audit_record.get('documentkey')
        field_name = audit_record.get('fieldname')
        old_value = audit_record.get('oldvalue')
        new_value = audit_record.get('newvalue')
        
        # Skip records without required fields
        if not all([table_name, document_key, field_name]):
            raise ValueError(f"Audit record {i} missing required fields: {audit_record}")
            
        # Initialize table in merged_state if it doesn't exist
        if table_name not in merged_state:
            merged_state[table_name] = {}
            
        # Initialize document in table if it doesn't exist
        if document_key not in merged_state[table_name]:
            merged_state[table_name][document_key] = {}
            
        # Handle different types of changes
        if new_value == "DELETE":
            # Record is being deleted - remove the entire document
            if document_key in merged_state[table_name]:
                del merged_state[table_name][document_key]
        else:
            # For all other cases (new fields, updates), just set to new_value
            # The audit records are processed in chronological order, so new_value
            # represents the state after this change
            merged_state[table_name][document_key][field_name] = new_value

    return merged_state 

if __name__ == "__main__":
    # Toggle auditing for all tables  
    # from time import perf_counter
    # start_time = perf_counter()
    # setup_auditing() 
    # print(f"Time taken to setup auditing: {perf_counter() - start_time} seconds")

    toggle_audit_inserts() # Enables auditing insertions to audited tables

    all_tables = get_all_tables(return_sys_tables=True) 
    all_tables_without_sys = get_all_tables(return_sys_tables=False)

    print(f"Number of all tables: {len(all_tables)}")
    print(f"Number of all tables without sys: {len(all_tables_without_sys)}")

    # Toggle on auditing for all non sys tables (difference between all_tables and all_tables_without_sys)
    system_tables = set(all_tables) - set(all_tables_without_sys)
    print(f"Number of system tables: {len(system_tables)}")

    for table in tqdm(all_tables_without_sys, desc="Toggling on auditing for non-system tables"):
        toggle_auditing_for_table(table, toggle_to=True)

    # Toggle off auditing for all system tables
    for table in tqdm(system_tables, desc="Toggling off auditing for system tables"):
        toggle_auditing_for_table(table, toggle_to=False)