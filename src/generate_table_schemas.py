import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
from pathlib import Path
from tqdm import tqdm
from rest_apis.world_model_scripts.states import get_all_tables
from browsergym.workarena.instance import SNowInstance
from src.browsergym.workarena.api.utils import table_api_call 
import re

def fetch_javascript_default_value(instance, js_default_value): 
    """
    Parse the javascript default value function and extract all function/class names used.
    Returns a list of external dependencies (classes/functions) that need to be looked up.
    """
    js_default_value = js_default_value.replace("javascript:", "").strip()
    
    # Remove comments to avoid picking up identifiers from comments
    # Remove single-line comments (// ...)
    js_default_value = re.sub(r'//.*?$', '', js_default_value, flags=re.MULTILINE)
    # Remove multi-line comments (/* ... */)
    js_default_value = re.sub(r'/\*.*?\*/', '', js_default_value, flags=re.DOTALL)
    
    external_dependencies = set()
    
    # 1. Extract script names from gs.include() calls
    include_pattern = r'gs\.include\s*\(\s*["\']([^"\']+)["\']\s*\)'
    includes = re.findall(include_pattern, js_default_value)
    external_dependencies.update(includes)
    
    # 2. Find Class.method patterns (external class usage)
    # Use negative lookbehind to avoid matching camelCase variables like changeRequestGr.method()
    class_method_pattern = r'(?<![a-z])([A-Z][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*'
    class_methods = re.findall(class_method_pattern, js_default_value)
    
    # 3. Find variables/classes being instantiated or used (starting with capital letter)
    # Look for patterns like "new ClassName" or standalone "ClassName" usage
    instantiation_pattern = r'\bnew\s+([A-Z][a-zA-Z0-9_]*)'
    instantiations = re.findall(instantiation_pattern, js_default_value)
    
    # Find standalone capital letter identifiers that are not being defined
    # Match complete words that start with a capital letter, not substrings
    # Use negative lookbehind to ensure the capital letter is at word boundary, not mid-word
    standalone_pattern = r'(?<![a-z])([A-Z][a-zA-Z0-9_]*)\b'
    all_capitals = re.findall(standalone_pattern, js_default_value)
    
    # Filter out those that are being defined in this script
    # Pattern 1: var/let/const/function declarations
    defined_pattern1 = r'\b(var|let|const|function)\s+([A-Z][a-zA-Z0-9_]*)'
    # Pattern 2: Assignment to function (e.g., MyClass = function)
    defined_pattern2 = r'\b([A-Z][a-zA-Z0-9_]*)\s*=\s*function'
    # Pattern 3: Prototype assignments that define new methods (e.g., MyClass.prototype = {...})
    defined_pattern3 = r'\b([A-Z][a-zA-Z0-9_]*)\.prototype\s*='
    # Pattern 4: Any assignment to a capitalized name (e.g., MyClass = Class.create())
    defined_pattern4 = r'\b([A-Z][a-zA-Z0-9_]*)\s*='
    
    defined_names = set()
    
    # Find all matches from pattern 1
    matches1 = re.findall(defined_pattern1, js_default_value)
    for match in matches1:
        defined_names.add(match[1])  # Second group is the class name
    
    # Find all matches from pattern 2
    matches2 = re.findall(defined_pattern2, js_default_value)
    defined_names.update(matches2)
    
    # Find all matches from pattern 3
    matches3 = re.findall(defined_pattern3, js_default_value)
    defined_names.update(matches3)
    
    # Find all matches from pattern 4
    matches4 = re.findall(defined_pattern4, js_default_value)
    defined_names.update(matches4)
    
    
    # Define built-ins to filter out
    js_builtins = {'Object', 'Array', 'String', 'Number', 'Boolean', 'Date', 'RegExp', 'Error', 'JSON', 'Math', 'Class'}
    
    def is_valid_external_dependency(name):
        """Check if a name should be considered an external dependency"""
        return (name not in defined_names and 
                name not in js_builtins and 
                not name.isupper())  # Filter out all-caps constants
    
    # Add class names from Class.method patterns (but exclude locally defined ones, built-ins, and constants)
    for class_name in class_methods:
        if is_valid_external_dependency(class_name):
            external_dependencies.add(class_name)
    
    # Add instantiated classes (but exclude locally defined ones, built-ins, and constants)
    for class_name in instantiations:
        if is_valid_external_dependency(class_name):
            external_dependencies.add(class_name)
    
    # Add standalone capital identifiers that are not defined locally, built-ins, or constants
    for capital in all_capitals:
        if is_valid_external_dependency(capital):
            external_dependencies.add(capital)
    
    # Remove common ServiceNow built-ins that we don't need to look up
    servicenow_builtins = {'GlideRecord', 'GlideElement', 'GlideDateTime', 'GlideUser', 'GlideSysAttachment'}
    external_dependencies = external_dependencies - servicenow_builtins
    
    return list(external_dependencies)

def generate_table_schemas():
    """
    Generate the table schemas for all tables in ServiceNow.
    """
    instance = SNowInstance()
    tables = get_all_tables(return_sys_tables=False) # Still includes sys_user, sys_user_group, sys_user_grmember, sys_user_has_role, sys_user_role
    params = {
        "sysparm_limit": 8000,
        "sysparm_fields": "reference, element, mandatory, internal_type, default_value",
    }
    out_file = Path(__file__).parent / "prompts" / "all_table_schemas.json"
    all_table_schemas = {}
    num_errors = 0 
    for table in tqdm(tables, desc="Getting table schemas"):
        try: 
            all_table_schemas[table] = []
            params["sysparm_query"] = f"name={table}"
            resp = table_api_call(instance, table="sys_dictionary", params=params) 

            # Check all the choices for the columns of the table 
            col_choices_resp = table_api_call(instance, table="sys_choice", params={
                "sysparm_query": f"name={table}", 
                "sysparm_fields": "value, element", 
                "sysparm_limit": 50,
            })
            if col_choices_resp['result']: 
                # Make dictionary of elements as keys and all its values as its list of values
                col_choices_dict = {}
                for col_choice in col_choices_resp['result']:
                    if col_choice['element'] not in col_choices_dict:
                        col_choices_dict[col_choice['element']] = []
                    col_choices_dict[col_choice['element']].append(col_choice['value'])

            for column_record in resp['result']:
                column_record['internal_type'] = column_record['internal_type'].get('value')

                # Check if column data type is time-field. If so, no need to add this to the schema. 
                if column_record['internal_type'] in ['glide_date_time', 'glide_duration', 'glide_time']: # Allowing glide_date because it's not bounded by time-precision 
                    continue 
                
                # Check if there are fixed choices for the column and add them to the column record
                if col_choices_resp['result']:
                    col_choices = col_choices_dict.get(column_record['element'], [])
                    if col_choices:
                        column_record['choices'] = json.dumps(col_choices)

                # Check if column's default value has javascript code; If so, get all external dependencies 
                # Iteratively build up the javascript code until no more external dependencies are found
                if 'javascript' in column_record['default_value']:
                    print(f"Processing javascript code for column {column_record['element']}: {column_record['default_value']}")
                    total_external_dependencies = set()
                    javascript_codes = column_record['default_value']
                    total_javascript_codes = javascript_codes
                    while True: 
                        external_dependencies = fetch_javascript_default_value(instance, javascript_codes)
                        external_dependencies = set(external_dependencies) - total_external_dependencies
                        if not external_dependencies:
                            break
                        total_external_dependencies.update(external_dependencies)
                        # Update javascript_code with script of all external dependencies  
                        js_params = {"sysparm_query": f"nameIN{','.join(external_dependencies)}"}
                        javascript_codes = table_api_call(instance, table="sys_script_include", params=js_params)['result']
                        javascript_codes = "\n".join([js['script'] for js in javascript_codes])
                        total_javascript_codes += "\n" + javascript_codes

                    print(f"Total javascript code for column {column_record['element']} in table {table}: {total_javascript_codes}")
                    column_record['all_javascript_context'] = total_javascript_codes             
                    
                all_table_schemas[table].append(column_record)
        except Exception as e:
            print(f"Error getting table schema for table {table}: {e}")
            num_errors += 1
            continue
    print(f"Number of errors: {num_errors}")

    with open(out_file, "w") as f:
        json.dump(all_table_schemas, f, indent=2)

if __name__ == "__main__":
    generate_table_schemas()