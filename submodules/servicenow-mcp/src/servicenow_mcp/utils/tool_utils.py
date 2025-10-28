from typing import Any, Callable, Dict, Tuple, Type, Optional

# Import all necessary tool implementation functions and params models
# (This list needs to be kept complete and up-to-date)
from servicenow_mcp.tools.catalog_optimization import (
    OptimizationRecommendationsParams,
    UpdateCatalogItemParams,
)
from servicenow_mcp.tools.catalog_optimization import (
    get_optimization_recommendations as get_optimization_recommendations_tool,
)
from servicenow_mcp.tools.catalog_optimization import (
    update_catalog_item as update_catalog_item_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    CreateCatalogCategoryParams,
    DeleteCatalogCategoryParams,
    GetCatalogItemParams,
    ListCatalogCategoriesParams,
    ListCatalogItemsParams,
    MoveCatalogItemsParams,
    UpdateCatalogCategoryParams,
    OrderCatalogItemParams,
    CreateCatalogItemParams,
)
from servicenow_mcp.tools.catalog_tools import (
    create_catalog_item as create_catalog_item_tool,
    delete_catalog_category as delete_catalog_category_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    create_catalog_category as create_catalog_category_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    get_catalog_item as get_catalog_item_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    list_catalog_categories as list_catalog_categories_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    list_catalog_items as list_catalog_items_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    move_catalog_items as move_catalog_items_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    update_catalog_category as update_catalog_category_tool,
)
from servicenow_mcp.tools.catalog_tools import (
    order_catalog_item as order_catalog_item_tool,
)
from servicenow_mcp.tools.catalog_variables import (
    CreateCatalogItemVariableParams,
    ListCatalogItemVariablesParams,
    UpdateCatalogItemVariableParams,
)
from servicenow_mcp.tools.catalog_variables import (
    create_catalog_item_variable as create_catalog_item_variable_tool,
)
from servicenow_mcp.tools.catalog_variables import (
    list_catalog_item_variables as list_catalog_item_variables_tool,
)
from servicenow_mcp.tools.catalog_variables import (
    update_catalog_item_variable as update_catalog_item_variable_tool,
)
from servicenow_mcp.tools.change_tools import (
    AddChangeTaskParams,
    ApproveChangeParams,
    CreateChangeRequestParams,
    GetChangeRequestDetailsParams,
    ListChangeRequestsParams,
    RejectChangeParams,
    SubmitChangeForApprovalParams,
    UpdateChangeRequestParams,
)
from servicenow_mcp.tools.change_tools import (
    add_change_task as add_change_task_tool,
)
from servicenow_mcp.tools.change_tools import (
    approve_change as approve_change_tool,
)
from servicenow_mcp.tools.change_tools import (
    create_change_request as create_change_request_tool,
)
from servicenow_mcp.tools.change_tools import (
    get_change_request_details as get_change_request_details_tool,
)
from servicenow_mcp.tools.change_tools import (
    list_change_requests as list_change_requests_tool,
)
from servicenow_mcp.tools.change_tools import (
    reject_change as reject_change_tool,
)
from servicenow_mcp.tools.change_tools import (
    submit_change_for_approval as submit_change_for_approval_tool,
)
from servicenow_mcp.tools.change_tools import (
    update_change_request as update_change_request_tool,
)
from servicenow_mcp.tools.changeset_tools import (
    AddFileToChangesetParams,
    CommitChangesetParams,
    CreateChangesetParams,
    GetChangesetDetailsParams,
    ListChangesetsParams,
    PublishChangesetParams,
    UpdateChangesetParams,
)
from servicenow_mcp.tools.changeset_tools import (
    add_file_to_changeset as add_file_to_changeset_tool,
)
from servicenow_mcp.tools.changeset_tools import (
    commit_changeset as commit_changeset_tool,
)
from servicenow_mcp.tools.changeset_tools import (
    create_changeset as create_changeset_tool,
)
from servicenow_mcp.tools.changeset_tools import (
    get_changeset_details as get_changeset_details_tool,
)
from servicenow_mcp.tools.changeset_tools import (
    list_changesets as list_changesets_tool,
)
from servicenow_mcp.tools.changeset_tools import (
    publish_changeset as publish_changeset_tool,
)
from servicenow_mcp.tools.changeset_tools import (
    update_changeset as update_changeset_tool,
)
from servicenow_mcp.tools.incident_tools import (
    AddCommentParams,
    CreateIncidentParams,
    ListIncidentsParams,
    ResolveIncidentParams,
    UpdateIncidentParams,
)
from servicenow_mcp.tools.incident_tools import (
    add_comment as add_comment_tool,
)
from servicenow_mcp.tools.incident_tools import (
    create_incident as create_incident_tool,
)
from servicenow_mcp.tools.incident_tools import (
    list_incidents as list_incidents_tool,
)
from servicenow_mcp.tools.incident_tools import (
    resolve_incident as resolve_incident_tool,
)
from servicenow_mcp.tools.incident_tools import (
    update_incident as update_incident_tool,
)
from servicenow_mcp.tools.knowledge_base import (
    CreateArticleParams,
    CreateKnowledgeBaseParams,
    GetArticleParams,
    ListArticlesParams,
    ListKnowledgeBasesParams,
    PublishArticleParams,
    UpdateArticleParams,
)
from servicenow_mcp.tools.knowledge_base import (
    CreateCategoryParams as CreateKBCategoryParams,  # Aliased
)
from servicenow_mcp.tools.knowledge_base import (
    ListCategoriesParams as ListKBCategoriesParams,  # Aliased
)
from servicenow_mcp.tools.knowledge_base import (
    create_article as create_article_tool,
)
from servicenow_mcp.tools.knowledge_base import (
    # create_category aliased in function call
    create_knowledge_base as create_knowledge_base_tool,
)
from servicenow_mcp.tools.knowledge_base import (
    get_article as get_article_tool,
)
from servicenow_mcp.tools.knowledge_base import (
    list_articles as list_articles_tool,
)
from servicenow_mcp.tools.knowledge_base import (
    # list_categories aliased in function call
    list_knowledge_bases as list_knowledge_bases_tool,
)
from servicenow_mcp.tools.knowledge_base import (
    publish_article as publish_article_tool,
)
from servicenow_mcp.tools.knowledge_base import (
    update_article as update_article_tool,
)
from servicenow_mcp.tools.script_include_tools import (
    CreateScriptIncludeParams,
    DeleteScriptIncludeParams,
    GetScriptIncludeParams,
    ListScriptIncludesParams,
    ScriptIncludeResponse,
    UpdateScriptIncludeParams,
)
from servicenow_mcp.tools.script_include_tools import (
    create_script_include as create_script_include_tool,
)
from servicenow_mcp.tools.script_include_tools import (
    delete_script_include as delete_script_include_tool,
)
from servicenow_mcp.tools.script_include_tools import (
    get_script_include as get_script_include_tool,
)
from servicenow_mcp.tools.script_include_tools import (
    list_script_includes as list_script_includes_tool,
)
from servicenow_mcp.tools.script_include_tools import (
    update_script_include as update_script_include_tool,
)
from servicenow_mcp.tools.user_tools import (
    ListGroupMembersParams,
    AddGroupMembersParams,
    CreateGroupParams,
    CreateUserParams,
    GetUserParams,
    ListGroupsParams,
    ListUsersParams,
    RemoveGroupMembersParams,
    UpdateGroupParams,
    UpdateUserParams,
    ListGroupClearanceParams,
    ListUserClearanceParams,
    UpdateGroupClearanceParams,
    UpdateUserClearanceParams,
)
from servicenow_mcp.tools.user_tools import (
    list_group_members as list_group_members_tool,
    add_group_members as add_group_members_tool,
    list_users_clearance as list_users_clearance_tool,
    list_groups_clearance as list_groups_clearance_tool,
    update_user_clearance as update_user_clearance_tool,
    update_group_clearance as update_group_clearance_tool,
)
from servicenow_mcp.tools.user_tools import (
    create_group as create_group_tool,
)
from servicenow_mcp.tools.user_tools import (
    create_user as create_user_tool,
)
from servicenow_mcp.tools.user_tools import (
    get_user as get_user_tool,
)
from servicenow_mcp.tools.user_tools import (
    list_groups as list_groups_tool,
)
from servicenow_mcp.tools.user_tools import (
    list_users as list_users_tool,
)
from servicenow_mcp.tools.user_tools import (
    remove_group_members as remove_group_members_tool,
)
from servicenow_mcp.tools.user_tools import (
    update_group as update_group_tool,
)
from servicenow_mcp.tools.user_tools import (
    update_user as update_user_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    ActivateWorkflowParams,
    AddWorkflowActivityParams,
    CreateWorkflowParams,
    DeactivateWorkflowParams,
    DeleteWorkflowParams,
    DeleteWorkflowActivityParams,
    GetWorkflowActivitiesParams,
    GetWorkflowDetailsParams,
    ListWorkflowsParams,
    ListWorkflowVersionsParams,
    ReorderWorkflowActivitiesParams,
    UpdateWorkflowActivityParams,
    UpdateWorkflowParams,
)
from servicenow_mcp.tools.workflow_tools import (
    activate_workflow as activate_workflow_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    add_workflow_activity as add_workflow_activity_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    create_workflow as create_workflow_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    deactivate_workflow as deactivate_workflow_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    delete_workflow_activity as delete_workflow_activity_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    delete_workflow as delete_workflow_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    get_workflow_activities as get_workflow_activities_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    get_workflow_details as get_workflow_details_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    list_workflow_versions as list_workflow_versions_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    list_workflows as list_workflows_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    reorder_workflow_activities as reorder_workflow_activities_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    update_workflow as update_workflow_tool,
)
from servicenow_mcp.tools.workflow_tools import (
    update_workflow_activity as update_workflow_activity_tool,
)
from servicenow_mcp.tools.story_tools import (
    CreateStoryParams,
    UpdateStoryParams,
    ListStoriesParams,
    ListStoryDependenciesParams,
    CreateStoryDependencyParams,
    DeleteStoryDependencyParams,
)
from servicenow_mcp.tools.story_tools import (
    create_story as create_story_tool,
    update_story as update_story_tool,
    list_stories as list_stories_tool,
    list_story_dependencies as list_story_dependencies_tool,
    create_story_dependency as create_story_dependency_tool,
    delete_story_dependency as delete_story_dependency_tool,
)
from servicenow_mcp.tools.epic_tools import (
    CreateEpicParams,
    UpdateEpicParams,
    ListEpicsParams,
)
from servicenow_mcp.tools.epic_tools import (
    create_epic as create_epic_tool,
    update_epic as update_epic_tool,
    list_epics as list_epics_tool,
)
from servicenow_mcp.tools.scrum_task_tools import (
    CreateScrumTaskParams,
    UpdateScrumTaskParams,
    ListScrumTasksParams,
)
from servicenow_mcp.tools.scrum_task_tools import (
    create_scrum_task as create_scrum_task_tool,
    update_scrum_task as update_scrum_task_tool,
    list_scrum_tasks as list_scrum_tasks_tool,
)
from servicenow_mcp.tools.project_tools import (
    CreateProjectParams,
    UpdateProjectParams,
    ListProjectsParams,
)
from servicenow_mcp.tools.project_tools import (
    create_project as create_project_tool,
    update_project as update_project_tool,
    list_projects as list_projects_tool,
)
from servicenow_mcp.tools.asset_tools import (
    CreateCurrencyInstanceParams,
    CreateAssetParams,
    UpdateAssetParams,
    GetAssetsParams,
    DeleteAssetParams,
    TransferAssetParams,
    ListHardwareAssetsParams,
    CreateHardwareAssetParams,
    UpdateHardwareAssetParams,
)
from servicenow_mcp.tools.asset_tools import (
    create_currency_instance as create_currency_instance_tool,
    create_asset as create_asset_tool,
    update_asset as update_asset_tool,
    get_assets as get_assets_tool,
    delete_asset as delete_asset_tool,
    transfer_asset as transfer_asset_tool,
    list_hardware_assets as list_hardware_assets_tool,
    create_hardware_asset as create_hardware_asset_tool,
    update_hardware_asset as update_hardware_asset_tool,
)
from servicenow_mcp.tools.record_tools import (
    CreateProblemParams,
)
from servicenow_mcp.tools.record_tools import (
    create_problem as create_problem_tool,
)
from servicenow_mcp.tools.request_tools import (
    CreateItemRequestParams,
    ListItemRequestsParams,
    ChangeRequestItemPriorityParams,
)
from servicenow_mcp.tools.request_tools import (
    create_item_request as create_item_request_tool,
    list_item_requests as list_item_requests_tool,
    change_request_item_priority as change_request_item_priority_tool,
)

from servicenow_mcp.tools.expense_tools import (
    ListExpenseLineParams,
    DeleteExpenseLineParams,
)
from servicenow_mcp.tools.expense_tools import (
    list_expense_lines as list_expense_lines_tool,
    delete_expense_line as delete_expense_line_tool,
)
from servicenow_mcp.tools.report_tools import (
    GetReportParams,
    GetReportIdsFromPortalWidgetsParams,
    GetCanvasParams,
    GetPortalWidgetsParams,
    GetDashboardTabParams,
    GetAnyTableParams,
)
from servicenow_mcp.tools.report_tools import (
    get_report as get_report_tool,
    get_report_ids_from_portal_widgets as get_report_ids_from_portal_widgets_tool,
    get_canvas as get_canvas_tool,
    get_portal_widgets as get_portal_widgets_tool,
    get_dashboard_tab as get_dashboard_tab_tool,
    search_any_table as search_any_table_tool,
)

from servicenow_mcp.tools.schema_tools import (
    GetTableSchemaParams,
)
from servicenow_mcp.tools.schema_tools import (
    get_table_schema as get_table_schema_tool,
)

from servicenow_mcp.auth.auth_manager import AuthManager
from servicenow_mcp.utils.config import ServerConfig
import requests
import logging
logger = logging.getLogger(__name__)

# Define a type alias for the Pydantic models or dataclasses used for params
ParamsModel = Type[Any]  # Use Type[Any] for broader compatibility initially

# Define the structure of the tool definition tuple
ToolDefinition = Tuple[
    Callable,  # Implementation function
    ParamsModel,  # Pydantic model for parameters
    Type,  # Return type annotation (used for hints, not strictly enforced by low-level server)
    str,  # Description
    str,  # Serialization method ('str', 'json', 'dict', 'model_dump', etc.)
]


def get_tool_definitions(
    create_kb_category_tool_impl: Callable, list_kb_categories_tool_impl: Callable
) -> Dict[str, ToolDefinition]:
    """
    Returns a dictionary containing definitions for all available ServiceNow tools.

    This centralizes the tool definitions for use in the server implementation.
    Pass aliased functions for KB categories directly.

    Returns:
        Dict[str, ToolDefinition]: A dictionary mapping tool names to their definitions.
    """
    tool_definitions: Dict[str, ToolDefinition] = {
        # Incident Tools
        "create_incident": (
            create_incident_tool,
            CreateIncidentParams,
            str,
            "Create a new incident in ServiceNow",
            "str",
        ),
        "update_incident": (
            update_incident_tool,
            UpdateIncidentParams,
            str,
            "Update an existing incident in ServiceNow",
            "str",
        ),
        "add_comment": (
            add_comment_tool,
            AddCommentParams,
            str,
            "Add a comment to an incident in ServiceNow",
            "str",
        ),
        "resolve_incident": (
            resolve_incident_tool,
            ResolveIncidentParams,
            str,
            "Resolve an incident in ServiceNow",
            "str",
        ),
        "list_incidents": (
            list_incidents_tool,
            ListIncidentsParams,
            str,  # Expects JSON string
            "List incidents from ServiceNow",
            "json",  # Tool returns list/dict, needs JSON dump
        ),
        # Catalog Tools
        "list_catalog_items": (
            list_catalog_items_tool,
            ListCatalogItemsParams,
            str,  # Expects JSON string
            "List service catalog items.",
            "json",  # Tool returns list/dict
        ),
        "create_catalog_item": (
            create_catalog_item_tool,
            CreateCatalogItemParams,
            str,  # Expects JSON string
            "Create a new service catalog item.",
            "json_dict",  # Tool returns Pydantic model
        ),
        "get_catalog_item": (
            get_catalog_item_tool,
            GetCatalogItemParams,
            str,  # Expects JSON string
            "Get a specific service catalog item.",
            "json_dict",  # Tool returns Pydantic model
        ),
        "list_catalog_categories": (
            list_catalog_categories_tool,
            ListCatalogCategoriesParams,
            str,  # Expects JSON string
            "List service catalog categories.",
            "json",  # Tool returns list/dict
        ),
        "create_catalog_category": (
            create_catalog_category_tool,
            CreateCatalogCategoryParams,
            str,  # Expects JSON string
            "Create a new service catalog category.",
            "json_dict",  # Tool returns Pydantic model
        ),
        "delete_catalog_category": (
            delete_catalog_category_tool,
            DeleteCatalogCategoryParams,
            str,  # Expects JSON string
            "Delete an existing service catalog category.",
            "json_dict",  # Tool returns Pydantic model
        ),
        "update_catalog_category": (
            update_catalog_category_tool,
            UpdateCatalogCategoryParams,
            str,  # Expects JSON string
            "Update an existing service catalog category.",
            "json_dict",  # Tool returns Pydantic model
        ),
        "move_catalog_items": (
            move_catalog_items_tool,
            MoveCatalogItemsParams,
            str,  # Expects JSON string
            "Move catalog items to a different category.",
            "json_dict",  # Tool returns Pydantic model
        ),
        "get_optimization_recommendations": (
            get_optimization_recommendations_tool,
            OptimizationRecommendationsParams,
            str,  # Expects JSON string
            "Get optimization recommendations for the service catalog.",
            "json",  # Tool returns list/dict
        ),
        "update_catalog_item": (
            update_catalog_item_tool,
            UpdateCatalogItemParams,
            str,  # Expects JSON string
            "Update a service catalog item.",
            "json",  # Tool returns Pydantic model
        ),
        # Catalog Variables
        "create_catalog_item_variable": (
            create_catalog_item_variable_tool,
            CreateCatalogItemVariableParams,
            Dict[str, Any],  # Expects dict
            "Create a new catalog item variable",
            "dict",  # Tool returns Pydantic model
        ),
        "list_catalog_item_variables": (
            list_catalog_item_variables_tool,
            ListCatalogItemVariablesParams,
            Dict[str, Any],  # Expects dict
            "List catalog item variables",
            "dict",  # Tool returns Pydantic model
        ),
        "update_catalog_item_variable": (
            update_catalog_item_variable_tool,
            UpdateCatalogItemVariableParams,
            Dict[str, Any],  # Expects dict
            "Update a catalog item variable",
            "dict",  # Tool returns Pydantic model
        ),
        "order_catalog_item": (
            order_catalog_item_tool,
            OrderCatalogItemParams,
            Dict[str, Any],  # Expects dict
            "Order a catalog item",
            "dict",  # Tool returns Pydantic model
        ),
        "change_request_item_priority": (
            change_request_item_priority_tool,
            ChangeRequestItemPriorityParams,
            Dict[str, Any],  # Expects dict
            "Change the priority of a change request item",
            "dict",  # Tool returns Pydantic model
        ),
        # Change Management Tools
        "create_change_request": (
            create_change_request_tool,
            CreateChangeRequestParams,
            str,
            "Create a new change request in ServiceNow",
            "str",
        ),
        "update_change_request": (
            update_change_request_tool,
            UpdateChangeRequestParams,
            str,
            "Update an existing change request in ServiceNow",
            "str",
        ),
        "list_change_requests": (
            list_change_requests_tool,
            ListChangeRequestsParams,
            str,  # Expects JSON string
            "List change requests from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        "get_change_request_details": (
            get_change_request_details_tool,
            GetChangeRequestDetailsParams,
            str,  # Expects JSON string
            "Get detailed information about a specific change request",
            "json",  # Tool returns list/dict
        ),
        "add_change_task": (
            add_change_task_tool,
            AddChangeTaskParams,
            str,  # Expects JSON string
            "Add a task to a change request",
            "json_dict",  # Tool returns Pydantic model
        ),
        "submit_change_for_approval": (
            submit_change_for_approval_tool,
            SubmitChangeForApprovalParams,
            str,
            "Submit a change request for approval",
            "str",  # Tool returns simple message
        ),
        "approve_change": (
            approve_change_tool,
            ApproveChangeParams,
            str,
            "Approve a change request",
            "str",  # Tool returns simple message
        ),
        "reject_change": (
            reject_change_tool,
            RejectChangeParams,
            str,
            "Reject a change request",
            "str",  # Tool returns simple message
        ),
        # Workflow Management Tools
        "list_workflows": (
            list_workflows_tool,
            ListWorkflowsParams,
            str,  # Expects JSON string
            "List workflows from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        "get_workflow_details": (
            get_workflow_details_tool,
            GetWorkflowDetailsParams,
            str,  # Expects JSON string
            "Get detailed information about a specific workflow",
            "json",  # Tool returns list/dict
        ),
        "list_workflow_versions": (
            list_workflow_versions_tool,
            ListWorkflowVersionsParams,
            str,  # Expects JSON string
            "List workflow versions from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        "get_workflow_activities": (
            get_workflow_activities_tool,
            GetWorkflowActivitiesParams,
            str,  # Expects JSON string
            "Get activities for a specific workflow",
            "json",  # Tool returns list/dict
        ),
        "create_workflow": (
            create_workflow_tool,
            CreateWorkflowParams,
            str,  # Expects JSON string
            "Create a new workflow in ServiceNow",
            "json_dict",  # Tool returns Pydantic model
        ),
        "update_workflow": (
            update_workflow_tool,
            UpdateWorkflowParams,
            str,  # Expects JSON string
            "Update an existing workflow in ServiceNow",
            "json_dict",  # Tool returns Pydantic model
        ),
        "activate_workflow": (
            activate_workflow_tool,
            ActivateWorkflowParams,
            str,
            "Activate a workflow in ServiceNow",
            "str",  # Tool returns simple message
        ),
        "deactivate_workflow": (
            deactivate_workflow_tool,
            DeactivateWorkflowParams,
            str,
            "Deactivate a workflow in ServiceNow",
            "str",  # Tool returns simple message
        ),
        "add_workflow_activity": (
            add_workflow_activity_tool,
            AddWorkflowActivityParams,
            str,  # Expects JSON string
            "Add a new activity to a workflow in ServiceNow",
            "json_dict",  # Tool returns Pydantic model
        ),
        "update_workflow_activity": (
            update_workflow_activity_tool,
            UpdateWorkflowActivityParams,
            str,  # Expects JSON string
            "Update an existing activity in a workflow",
            "json_dict",  # Tool returns Pydantic model
        ),
        "delete_workflow_activity": (
            delete_workflow_activity_tool,
            DeleteWorkflowActivityParams,
            str,
            "Delete an activity from a workflow",
            "str",  # Tool returns simple message
        ),
        "delete_workflow": (
            delete_workflow_tool,
            DeleteWorkflowParams,
            str,
            "Delete a workflow",
            "str",  # Tool returns simple message
        ),
        "reorder_workflow_activities": (
            reorder_workflow_activities_tool,
            ReorderWorkflowActivitiesParams,
            str,
            "Reorder activities in a workflow",
            "str",  # Tool returns simple message
        ),
        # Changeset Management Tools
        "list_changesets": (
            list_changesets_tool,
            ListChangesetsParams,
            str,  # Expects JSON string
            "List changesets from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        "get_changeset_details": (
            get_changeset_details_tool,
            GetChangesetDetailsParams,
            str,  # Expects JSON string
            "Get detailed information about a specific changeset",
            "json",  # Tool returns list/dict
        ),
        "create_changeset": (
            create_changeset_tool,
            CreateChangesetParams,
            str,  # Expects JSON string
            "Create a new changeset in ServiceNow",
            "json_dict",  # Tool returns Pydantic model
        ),
        "update_changeset": (
            update_changeset_tool,
            UpdateChangesetParams,
            str,  # Expects JSON string
            "Update an existing changeset in ServiceNow",
            "json_dict",  # Tool returns Pydantic model
        ),
        "commit_changeset": (
            commit_changeset_tool,
            CommitChangesetParams,
            str,
            "Commit a changeset in ServiceNow",
            "str",  # Tool returns simple message
        ),
        "publish_changeset": (
            publish_changeset_tool,
            PublishChangesetParams,
            str,
            "Publish a changeset in ServiceNow",
            "str",  # Tool returns simple message
        ),
        "add_file_to_changeset": (
            add_file_to_changeset_tool,
            AddFileToChangesetParams,
            str,
            "Add a file to a changeset in ServiceNow",
            "str",  # Tool returns simple message
        ),
        # Script Include Tools
        "list_script_includes": (
            list_script_includes_tool,
            ListScriptIncludesParams,
            Dict[str, Any],  # Expects dict
            "List script includes from ServiceNow",
            "raw_dict",  # Tool returns raw dict
        ),
        "get_script_include": (
            get_script_include_tool,
            GetScriptIncludeParams,
            Dict[str, Any],  # Expects dict
            "Get a specific script include from ServiceNow",
            "raw_dict",  # Tool returns raw dict
        ),
        "create_script_include": (
            create_script_include_tool,
            CreateScriptIncludeParams,
            ScriptIncludeResponse,  # Expects Pydantic model
            "Create a new script include in ServiceNow",
            "raw_pydantic",  # Tool returns Pydantic model
        ),
        "update_script_include": (
            update_script_include_tool,
            UpdateScriptIncludeParams,
            ScriptIncludeResponse,  # Expects Pydantic model
            "Update an existing script include in ServiceNow",
            "raw_pydantic",  # Tool returns Pydantic model
        ),
        "delete_script_include": (
            delete_script_include_tool,
            DeleteScriptIncludeParams,
            str,  # Expects JSON string
            "Delete a script include in ServiceNow",
            "json_dict",  # Tool returns Pydantic model
        ),
        # Knowledge Base Tools
        "create_knowledge_base": (
            create_knowledge_base_tool,
            CreateKnowledgeBaseParams,
            str,  # Expects JSON string
            "Create a new knowledge base in ServiceNow",
            "json_dict",  # Tool returns Pydantic model
        ),
        "list_knowledge_bases": (
            list_knowledge_bases_tool,
            ListKnowledgeBasesParams,
            Dict[str, Any],  # Expects dict
            "List knowledge bases from ServiceNow",
            "raw_dict",  # Tool returns raw dict
        ),
        # Use the passed-in implementations for aliased KB category tools
        "create_category": (
            create_kb_category_tool_impl,  # Use passed function
            CreateKBCategoryParams,
            str,  # Expects JSON string
            "Create a new category in a knowledge base",
            "json_dict",  # Tool returns Pydantic model
        ),
        "create_article": (
            create_article_tool,
            CreateArticleParams,
            str,  # Expects JSON string
            "Create a new knowledge article",
            "json_dict",  # Tool returns Pydantic model
        ),
        "update_article": (
            update_article_tool,
            UpdateArticleParams,
            str,  # Expects JSON string
            "Update an existing knowledge article",
            "json_dict",  # Tool returns Pydantic model
        ),
        "publish_article": (
            publish_article_tool,
            PublishArticleParams,
            str,  # Expects JSON string
            "Publish a knowledge article",
            "json_dict",  # Tool returns Pydantic model
        ),
        "list_articles": (
            list_articles_tool,
            ListArticlesParams,
            Dict[str, Any],  # Expects dict
            "List knowledge articles",
            "raw_dict",  # Tool returns raw dict
        ),
        "get_article": (
            get_article_tool,
            GetArticleParams,
            Dict[str, Any],  # Expects dict
            "Get a specific knowledge article by ID",
            "raw_dict",  # Tool returns raw dict
        ),
        # Use the passed-in implementations for aliased KB category tools
        "list_categories": (
            list_kb_categories_tool_impl,  # Use passed function
            ListKBCategoriesParams,
            Dict[str, Any],  # Expects dict
            "List categories in a knowledge base",
            "raw_dict",  # Tool returns raw dict
        ),
        # User Management Tools
        "list_users_clearance": (
            list_users_clearance_tool,
            ListUserClearanceParams,
            Dict[str, Any],  # Expects dict
            "List clearance level for users in ServiceNow",
            "raw_dict",
        ),
        "list_groups_clearance": (
            list_groups_clearance_tool,
            ListGroupClearanceParams,
            Dict[str, Any],  # Expects dict
            "List clearance level for groups in ServiceNow",
            "raw_dict",
        ),
        "update_group_clearance": (
            update_group_clearance_tool,
            UpdateGroupClearanceParams,
            Dict[str, Any],  # Expects dict
            "Update clearance level for a group in ServiceNow",
            "raw_dict",
        ),
        "update_user_clearance": (
            update_user_clearance_tool,
            UpdateUserClearanceParams,
            Dict[str, Any],  # Expects dict
            "Update clearance level for a user in ServiceNow",
            "raw_dict",
        ),
        "create_user": (
            create_user_tool,
            CreateUserParams,
            Dict[str, Any],  # Expects dict
            "Create a new user in ServiceNow",
            "raw_dict",  # Tool returns raw dict
        ),
        "update_user": (
            update_user_tool,
            UpdateUserParams,
            Dict[str, Any],  # Expects dict
            "Update an existing user in ServiceNow",
            "raw_dict",
        ),
        "get_user": (
            get_user_tool,
            GetUserParams,
            Dict[str, Any],  # Expects dict
            "Get a specific user in ServiceNow",
            "raw_dict",
        ),
        "list_users": (
            list_users_tool,
            ListUsersParams,
            Dict[str, Any],  # Expects dict
            "List users in ServiceNow",
            "raw_dict",
        ),
        "create_group": (
            create_group_tool,
            CreateGroupParams,
            Dict[str, Any],  # Expects dict
            "Create a new group in ServiceNow",
            "raw_dict",
        ),
        "update_group": (
            update_group_tool,
            UpdateGroupParams,
            Dict[str, Any],  # Expects dict
            "Update an existing group in ServiceNow",
            "raw_dict",
        ),
        "list_group_members": (
            list_group_members_tool,
            ListGroupMembersParams,
            Dict[str, Any],  # Expects dict
            "List members of an existing group in ServiceNow",
            "raw_dict",
        ),
        "add_group_members": (
            add_group_members_tool,
            AddGroupMembersParams,
            Dict[str, Any],  # Expects dict
            "Add members to an existing group in ServiceNow",
            "raw_dict",
        ),
        "remove_group_members": (
            remove_group_members_tool,
            RemoveGroupMembersParams,
            Dict[str, Any],  # Expects dict
            "Remove members from an existing group in ServiceNow",
            "raw_dict",
        ),
        "list_groups": (
            list_groups_tool,
            ListGroupsParams,
            Dict[str, Any],  # Expects dict
            "List groups from ServiceNow with optional filtering",
            "raw_dict",
        ),
        # Story Management Tools
        "create_story": (
            create_story_tool,
            CreateStoryParams,
            str,
            "Create a new story in ServiceNow",
            "str",
        ),
        "update_story": (
            update_story_tool,
            UpdateStoryParams,
            str,
            "Update an existing story in ServiceNow",
            "str",
        ),
        "list_stories": (
            list_stories_tool,
            ListStoriesParams,
            str,  # Expects JSON string
            "List stories from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        "list_story_dependencies": (
            list_story_dependencies_tool,
            ListStoryDependenciesParams,
            str,  # Expects JSON string
            "List story dependencies from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        "create_story_dependency": (
            create_story_dependency_tool,
            CreateStoryDependencyParams,
            str,
            "Create a dependency between two stories in ServiceNow",
            "str",
        ),
        "delete_story_dependency": (
            delete_story_dependency_tool,
            DeleteStoryDependencyParams,
            str,
            "Delete a story dependency in ServiceNow",
            "str",
        ),
        # Epic Management Tools
        "create_epic": (
            create_epic_tool,
            CreateEpicParams,
            str,
            "Create a new epic in ServiceNow",
            "str",
        ),
        "update_epic": (
            update_epic_tool,
            UpdateEpicParams,
            str,
            "Update an existing epic in ServiceNow",
            "str",
        ),
        "list_epics": (
            list_epics_tool,
            ListEpicsParams,
            str,  # Expects JSON string
            "List epics from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        # Scrum Task Management Tools
        "create_scrum_task": (
            create_scrum_task_tool,
            CreateScrumTaskParams,
            str,
            "Create a new scrum task in ServiceNow",
            "str",
        ),
        "update_scrum_task": (
            update_scrum_task_tool,
            UpdateScrumTaskParams,
            str,
            "Update an existing scrum task in ServiceNow",
            "str",
        ),
        "list_scrum_tasks": (
            list_scrum_tasks_tool,
            ListScrumTasksParams,
            str,  # Expects JSON string
            "List scrum tasks from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        # Project Management Tools
        "create_project": (
            create_project_tool,
            CreateProjectParams,
            str,
            "Create a new project in ServiceNow",
            "str",
        ),
        "update_project": (
            update_project_tool,
            UpdateProjectParams,
            str,
            "Update an existing project in ServiceNow",
            "str",
        ),
        "list_projects": (
            list_projects_tool,
            ListProjectsParams,
            str,  # Expects JSON string
            "List projects from ServiceNow",
            "json",  # Tool returns list/dict
        ),
        # Asset Management Tools
        "create_asset": (
            create_asset_tool,
            CreateAssetParams,
            str,
            "Create a new asset in ServiceNow",
            "str",
        ),
        "create_currency_instance": (
            create_currency_instance_tool,
            CreateCurrencyInstanceParams,
            Dict[str, Any],
            "Create a new currency instance in ServiceNow",
            "raw_dict",
        ),
        "update_asset": (
            update_asset_tool,
            UpdateAssetParams,
            str,
            "Update an existing asset in ServiceNow",
            "str",
        ),
        "get_assets": (
            get_assets_tool,
            GetAssetsParams,
            Dict[str, Any],  # Expects dict
            "Get, list, or search for assets in ServiceNow. Supports single asset lookup by ID/tag/serial, filtering by user/location, and searching by name or general query.",
            "raw_dict",
        ),
        "delete_asset": (
            delete_asset_tool,
            DeleteAssetParams,
            str,
            "Delete an asset from ServiceNow",
            "str",
        ),
        "transfer_asset": (
            transfer_asset_tool,
            TransferAssetParams,
            str,
            "Transfer an asset to a different user in ServiceNow",
            "str",
        ),
        # Problem Management Tools
        "create_problem": (
            create_problem_tool,
            CreateProblemParams,
            str,
            "Create a new problem in ServiceNow",
            "str",
        ),
        # Hardware Asset Management Tools
        "list_hardware_assets": (
            list_hardware_assets_tool,
            ListHardwareAssetsParams,
            Dict[str, Any],  # Expects dict
            "List hardware assets from ServiceNow",
            "raw_dict",
        ),
        "create_hardware_asset": (
            create_hardware_asset_tool,
            CreateHardwareAssetParams,
            Dict[str,Any],
            "Create a new hardware asset in ServiceNow",
            "str",
        ),
        "update_hardware_asset": (
            update_hardware_asset_tool,
            UpdateHardwareAssetParams,
            Dict[str, Any],
            "Update an existing hardware asset in ServiceNow",
            "raw_dict",
        ),
        # Request Management Tools
        "create_item_request": (
            create_item_request_tool,
            CreateItemRequestParams,
            str,
            "Create a new item request in ServiceNow. This is used to create a request for a specific item. You can link multiple item requests to a single request object.",
            "str",
        ),
        "list_item_requests": (
            list_item_requests_tool,
            ListItemRequestsParams,
            Dict[str, Any],
            "List item requests from ServiceNow",
            "raw_dict",
        ),

        # Expense Management Tools
        "list_expense_lines": (
            list_expense_lines_tool,
            ListExpenseLineParams,
            Dict[str, Any],
            "List expense lines from ServiceNow. These hold the details of all the expenses that are made.", 
            "raw_dict",
        ),
        "delete_expense_line": (
            delete_expense_line_tool,
            DeleteExpenseLineParams,
            dict,
            "Delete an expense line from ServiceNow",
            "dict",
        ),

        # Report Management Tools
        "get_report": (
            get_report_tool,
            GetReportParams,
            Dict[str, Any],
            "Get a specific report from ServiceNow. All the ServiceNow charts are represented as reports.",
            "raw_dict",
        ),
        "get_report_ids_from_portal_widgets": (
            get_report_ids_from_portal_widgets_tool,
            GetReportIdsFromPortalWidgetsParams,
            Dict[str, Any],
            "Get all the report ids linked to the portal widgets on a dashboard.",
            "raw_dict",
        ), 
        "get_canvas": (
            get_canvas_tool,
            GetCanvasParams,
            Dict[str, Any],
            "Get the canvas page id linked to the dashboard tab.",
            "raw_dict",
        ),
        "get_portal_widgets": (
            get_portal_widgets_tool,
            GetPortalWidgetsParams,
            Dict[str, Any],
            "Get all the portal widget ids linked to the canvas page.",
            "raw_dict",
        ),
        "get_dashboard_tab": (
            get_dashboard_tab_tool,
            GetDashboardTabParams,
            Dict[str, Any],
            "Get the dashboard tab id linked to the dashboard.",
            "raw_dict",
        ),
        "search_any_table": (
            search_any_table_tool,
            GetAnyTableParams,
            Dict[str, Any],
            "Search any ServiceNow table.",
            "raw_dict",
        ),
        "get_table_schema": (
            get_table_schema_tool,
            GetTableSchemaParams,
            Dict[str, Any],
            "Get the schema of a ServiceNow table.",
            "raw_dict",
        ),
    }
    return tool_definitions
