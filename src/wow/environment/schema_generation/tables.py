import json
import re
from pathlib import Path
from typing import List
from tqdm import tqdm

from ...instance import SNowInstance
from ...api.utils import table_api_call
from ..states import get_all_tables


def fetch_javascript_default_value(js_default_value: str) -> List[str]:
    """
    Parse the javascript default value function and extract all function/class names used.
    Returns a list of external dependencies (classes/functions) that need to be looked up.
    """
    js_default_value = js_default_value.replace("javascript:", "").strip()

    js_default_value = re.sub(r'//.*?$', '', js_default_value, flags=re.MULTILINE)
    js_default_value = re.sub(r'/\*.*?\*/', '', js_default_value, flags=re.DOTALL)

    external_dependencies = set()

    include_pattern = r'gs\.include\s*\(\s*["\']([^"\']+)["\']\s*\)'
    includes = re.findall(include_pattern, js_default_value)
    external_dependencies.update(includes)

    class_method_pattern = r'(?<![a-z])([A-Z][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*'
    class_methods = re.findall(class_method_pattern, js_default_value)

    instantiation_pattern = r'\bnew\s+([A-Z][a-zA-Z0-9_]*)'
    instantiations = re.findall(instantiation_pattern, js_default_value)

    standalone_pattern = r'(?<![a-z])([A-Z][a-zA-Z0-9_]*)\b'
    all_capitals = re.findall(standalone_pattern, js_default_value)

    defined_pattern1 = r'\b(var|let|const|function)\s+([A-Z][a-zA-Z0-9_]*)'
    defined_pattern2 = r'\b([A-Z][a-zA-Z0-9_]*)\s*=\s*function'
    defined_pattern3 = r'\b([A-Z][a-zA-Z0-9_]*)\.prototype\s*='
    defined_pattern4 = r'\b([A-Z][a-zA-Z0-9_]*)\s*='

    defined_names = set()

    matches1 = re.findall(defined_pattern1, js_default_value)
    for match in matches1:
        defined_names.add(match[1])

    matches2 = re.findall(defined_pattern2, js_default_value)
    defined_names.update(matches2)

    matches3 = re.findall(defined_pattern3, js_default_value)
    defined_names.update(matches3)

    matches4 = re.findall(defined_pattern4, js_default_value)
    defined_names.update(matches4)

    js_builtins = {'Object', 'Array', 'String', 'Number', 'Boolean', 'Date', 'RegExp', 'Error', 'JSON', 'Math', 'Class'}

    def is_valid_external_dependency(name):
        return (name not in defined_names and
                name not in js_builtins and
                not name.isupper())

    for class_name in class_methods:
        if is_valid_external_dependency(class_name):
            external_dependencies.add(class_name)

    for class_name in instantiations:
        if is_valid_external_dependency(class_name):
            external_dependencies.add(class_name)

    for capital in all_capitals:
        if is_valid_external_dependency(capital):
            external_dependencies.add(capital)

    servicenow_builtins = {'GlideRecord', 'GlideElement', 'GlideDateTime', 'GlideUser', 'GlideSysAttachment'}
    external_dependencies = external_dependencies - servicenow_builtins

    return list(external_dependencies)


def generate_table_schemas(output_file: str = None):
    """
    Generate the table schemas for all tables in ServiceNow.
    """
    instance = SNowInstance()
    tables = get_all_tables(return_sys_tables=False)
    params = {
        "sysparm_limit": 8000,
        "sysparm_fields": "reference, element, mandatory, internal_type, default_value",
    }

    if output_file is None:
        base_dir = Path(__file__).parent.parent
        output_file = base_dir / "prompts" / "all_table_schemas.json"
    else:
        output_file = Path(output_file)

    all_table_schemas = {}
    num_errors = 0

    for table in tqdm(tables, desc="Getting table schemas"):
        try:
            all_table_schemas[table] = []
            params["sysparm_query"] = f"name={table}"
            resp = table_api_call(instance, table="sys_dictionary", params=params)

            col_choices_resp = table_api_call(instance, table="sys_choice", params={
                "sysparm_query": f"name={table}",
                "sysparm_fields": "value, element",
                "sysparm_limit": 50,
            })

            col_choices_dict = {}
            if col_choices_resp['result']:
                for col_choice in col_choices_resp['result']:
                    if col_choice['element'] not in col_choices_dict:
                        col_choices_dict[col_choice['element']] = []
                    col_choices_dict[col_choice['element']].append(col_choice['value'])

            for column_record in resp['result']:
                column_record['internal_type'] = column_record['internal_type'].get('value')

                if column_record['internal_type'] in ['glide_date_time', 'glide_duration', 'glide_time']:
                    continue

                if col_choices_resp['result']:
                    col_choices = col_choices_dict.get(column_record['element'], [])
                    if col_choices:
                        column_record['choices'] = json.dumps(col_choices)

                if 'javascript' in column_record['default_value']:
                    total_external_dependencies = set()
                    javascript_codes = column_record['default_value']
                    total_javascript_codes = javascript_codes

                    while True:
                        external_dependencies = fetch_javascript_default_value(javascript_codes)
                        external_dependencies = set(external_dependencies) - total_external_dependencies
                        if not external_dependencies:
                            break
                        total_external_dependencies.update(external_dependencies)

                        js_params = {"sysparm_query": f"nameIN{','.join(external_dependencies)}"}
                        javascript_codes = table_api_call(instance, table="sys_script_include", params=js_params)['result']
                        javascript_codes = "\n".join([js['script'] for js in javascript_codes])
                        total_javascript_codes += "\n" + javascript_codes

                    column_record['all_javascript_context'] = total_javascript_codes

                all_table_schemas[table].append(column_record)

        except Exception as e:
            print(f"Error getting table schema for table {table}: {e}")
            num_errors += 1
            continue

    print(f"Number of errors: {num_errors}")

    with open(output_file, "w") as f:
        json.dump(all_table_schemas, f, indent=2)

    print(f"Saved table schemas to: {output_file}")
    return all_table_schemas
