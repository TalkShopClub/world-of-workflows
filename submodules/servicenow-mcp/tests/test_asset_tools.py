"""
Tests for ServiceNow asset management tools.
"""

import json
import unittest
from unittest.mock import Mock, patch

import requests
import os

from servicenow_mcp.auth.auth_manager import AuthManager
from servicenow_mcp.tools.asset_tools import (
    CreateAssetParams,
    UpdateAssetParams,
    GetAssetParams,
    ListAssetsParams,
    ListHardwareAssetsParams,
    DeleteAssetParams,
    TransferAssetParams,
    SearchAssetsByNameParams,
    create_asset,
    update_asset,
    get_asset,
    list_assets,
    list_hardware_assets,
    delete_asset,
    transfer_asset,
    search_assets_by_name,
)
from servicenow_mcp.utils.resolvers import resolve_user_id, resolve_asset_id
from servicenow_mcp.utils.config import ServerConfig


class TestAssetTools(unittest.TestCase):
    """Test cases for asset management tools."""

    def setUp(self):
        """Set up test fixtures."""
        instance_url = os.environ.get("SNOW_INSTANCE_URL")
        api_url = f"{instance_url}/api/now"
        self.config = ServerConfig(
            instance_url=instance_url,
            api_url=api_url,
            auth={"type": "basic", "username": os.environ.get("SNOW_INSTANCE_UNAME"), "password": os.environ.get("SNOW_INSTANCE_PWD")},
            timeout=30,
        )
        self.auth_manager = Mock(spec=AuthManager)
        self.auth_manager.get_headers.return_value = {"Authorization": "Basic test"}

    @patch("servicenow_mcp.tools.asset_tools.requests.post")
    def test_create_asset_success(self, mock_post):
        """Test successful asset creation."""
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "sys_id": "test_asset_id",
                "asset_tag": "ASSET001",
                "display_name": "Test Laptop",
            }
        }
        mock_post.return_value = mock_response

        # Test parameters
        params = CreateAssetParams(
            asset_tag="ASSET001",
            display_name="Test Laptop",
            model="Dell XPS 13",
            serial_number="SN123456",
            cost="1500",
        )

        # Call function
        result = create_asset(self.config, self.auth_manager, params)

        # Assertions
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Asset created successfully")
        self.assertEqual(result.asset_id, "test_asset_id")
        self.assertEqual(result.asset_tag, "ASSET001")

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[0][0], f"{self.config.api_url}/table/alm_asset")
        
        expected_data = {
            "asset_tag": "ASSET001",
            "display_name": "Test Laptop",
            "model": "Dell XPS 13",
            "serial_number": "SN123456",
            "cost": "1500",
        }
        self.assertEqual(call_args[1]["json"], expected_data)

    @patch("servicenow_mcp.tools.asset_tools.requests.post")
    def test_create_asset_with_user_assignment(self, mock_post):
        """Test asset creation with user assignment."""
        # Mock asset creation response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "sys_id": "test_asset_id",
                "asset_tag": "ASSET001",
            }
        }
        mock_post.return_value = mock_response

        # Mock user resolution
        with patch("servicenow_mcp.tools.asset_tools._resolve_user_id") as mock_resolve:
            mock_resolve.return_value = "user_sys_id"

            params = CreateAssetParams(
                asset_tag="ASSET001",
                display_name="Test Laptop",
                assigned_to="john.doe",
            )

            result = create_asset(self.config, self.auth_manager, params)

            self.assertTrue(result.success)
            mock_resolve.assert_called_once_with(self.config, self.auth_manager, "john.doe")

    @patch("servicenow_mcp.tools.asset_tools.requests.post")
    def test_create_asset_user_not_found(self, mock_post):
        """Test asset creation when assigned user is not found."""
        with patch("servicenow_mcp.tools.asset_tools._resolve_user_id") as mock_resolve:
            mock_resolve.return_value = None

            params = CreateAssetParams(
                asset_tag="ASSET001",
                display_name="Test Laptop",
                assigned_to="invalid_user",
            )

            result = create_asset(self.config, self.auth_manager, params)

            self.assertFalse(result.success)
            self.assertIn("Could not resolve user", result.message)

    @patch("servicenow_mcp.tools.asset_tools.requests.post")
    def test_create_asset_api_error(self, mock_post):
        """Test asset creation with API error."""
        mock_post.side_effect = requests.RequestException("API Error")

        params = CreateAssetParams(
            asset_tag="ASSET001",
            display_name="Test Laptop",
        )

        result = create_asset(self.config, self.auth_manager, params)

        self.assertFalse(result.success)
        self.assertIn("Failed to create asset", result.message)

    @patch("servicenow_mcp.tools.asset_tools._resolve_asset_id")
    @patch("servicenow_mcp.tools.asset_tools.requests.patch")
    def test_update_asset_success(self, mock_patch, mock_resolve_asset):
        """Test successful asset update."""
        # Mock asset resolution
        mock_resolve_asset.return_value = "asset_sys_id"

        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "sys_id": "asset_sys_id",
                "asset_tag": "ASSET001",
            }
        }
        mock_patch.return_value = mock_response

        params = UpdateAssetParams(
            asset_id="ASSET001",
            display_name="Updated Laptop",
            cost="1600",
        )

        result = update_asset(self.config, self.auth_manager, params)

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Asset updated successfully")
        mock_resolve_asset.assert_called_once_with(self.config, self.auth_manager, "ASSET001")

    @patch("servicenow_mcp.tools.asset_tools._resolve_asset_id")
    def test_update_asset_not_found(self, mock_resolve_asset):
        """Test asset update when asset is not found."""
        mock_resolve_asset.return_value = None

        params = UpdateAssetParams(
            asset_id="INVALID_ASSET",
            display_name="Updated Laptop",
        )

        result = update_asset(self.config, self.auth_manager, params)

        self.assertFalse(result.success)
        self.assertIn("Could not find asset", result.message)

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_get_asset_by_id_success(self, mock_get):
        """Test successful asset retrieval by sys_id."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [{
                "sys_id": "asset_sys_id",
                "asset_tag": "ASSET001",
                "display_name": "Test Laptop",
            }]
        }
        mock_get.return_value = mock_response

        params = GetAssetParams(asset_id="asset_sys_id")
        result = get_asset(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["message"], "Asset found")
        self.assertIn("asset", result)

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_get_asset_by_tag_success(self, mock_get):
        """Test successful asset retrieval by asset tag."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [{
                "sys_id": "asset_sys_id",
                "asset_tag": "ASSET001",
                "display_name": "Test Laptop",
            }]
        }
        mock_get.return_value = mock_response

        params = GetAssetParams(asset_tag="ASSET001")
        result = get_asset(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        # Verify query parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn("sysparm_query", call_args[1]["params"])
        self.assertEqual(call_args[1]["params"]["sysparm_query"], "asset_tag=ASSET001")

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_get_asset_not_found(self, mock_get):
        """Test asset retrieval when asset is not found."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": []}
        mock_get.return_value = mock_response

        params = GetAssetParams(asset_id="nonexistent_id")
        result = get_asset(self.config, self.auth_manager, params)

        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Asset not found")

    def test_get_asset_no_params(self):
        """Test asset retrieval with no search parameters."""
        params = GetAssetParams()
        result = get_asset(self.config, self.auth_manager, params)

        self.assertFalse(result["success"])
        self.assertIn("At least one search parameter is required", result["message"])

    @patch("servicenow_mcp.tools.asset_tools._resolve_user_id")
    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_assets_with_filters(self, mock_get, mock_resolve_user):
        """Test asset listing with filters."""
        mock_resolve_user.return_value = "user_sys_id"
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "asset1",
                    "asset_tag": "ASSET001",
                    "display_name": "Laptop 1",
                },
                {
                    "sys_id": "asset2", 
                    "asset_tag": "ASSET002",
                    "display_name": "Laptop 2",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListAssetsParams(
            limit=20,
            assigned_to="john.doe",
            state="1",
            query="Laptop",
        )

        result = list_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertIn("assets", result)
        
        # Verify user resolution was called
        mock_resolve_user.assert_called_once_with(self.config, self.auth_manager, "john.doe")

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_assets_with_name_filter(self, mock_get):
        """Test asset listing with name filter."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "asset1",
                    "asset_tag": "LAPTOP001",
                    "display_name": "Dell XPS 13",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListAssetsParams(
            limit=10,
            name="Dell XPS",
        )

        result = list_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        
        # Verify the query includes name filter
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertIn("display_nameLIKEDell XPS", query_param)

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_search_assets_by_name_like_match(self, mock_get):
        """Test searching assets by name with LIKE matching."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "asset1",
                    "asset_tag": "LAPTOP001",
                    "display_name": "Dell XPS 13",
                },
                {
                    "sys_id": "asset2",
                    "asset_tag": "LAPTOP002", 
                    "display_name": "Dell XPS 15",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = SearchAssetsByNameParams(
            name="Dell XPS",
            limit=10,
            exact_match=False,
        )

        result = search_assets_by_name(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["search_term"], "Dell XPS")
        self.assertFalse(result["exact_match"])
        self.assertIn("assets", result)
        
        # Verify LIKE query was used
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertEqual(query_param, "display_nameLIKEDell XPS")

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_search_assets_by_name_exact_match(self, mock_get):
        """Test searching assets by name with exact matching."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "asset1",
                    "asset_tag": "LAPTOP001",
                    "display_name": "Dell XPS 13",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = SearchAssetsByNameParams(
            name="Dell XPS 13",
            limit=10,
            exact_match=True,
        )

        result = search_assets_by_name(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["search_term"], "Dell XPS 13")
        self.assertTrue(result["exact_match"])
        
        # Verify exact match query was used
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertEqual(query_param, "display_name=Dell XPS 13")

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_search_assets_by_name_no_results(self, mock_get):
        """Test searching assets by name with no results."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": []}
        mock_get.return_value = mock_response

        params = SearchAssetsByNameParams(
            name="Nonexistent Asset",
            limit=10,
        )

        result = search_assets_by_name(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)
        self.assertIn("Found 0 assets", result["message"])

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_search_assets_by_name_api_error(self, mock_get):
        """Test search assets by name with API error."""
        mock_get.side_effect = requests.RequestException("API Error")

        params = SearchAssetsByNameParams(
            name="Dell XPS",
            limit=10,
        )

        result = search_assets_by_name(self.config, self.auth_manager, params)

        self.assertFalse(result["success"])
        self.assertIn("Failed to search assets by name", result["message"])

    @patch("servicenow_mcp.tools.asset_tools._resolve_asset_id")
    @patch("servicenow_mcp.tools.asset_tools.requests.delete")
    def test_delete_asset_success(self, mock_delete, mock_resolve_asset):
        """Test successful asset deletion."""
        mock_resolve_asset.return_value = "asset_sys_id"
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_delete.return_value = mock_response

        params = DeleteAssetParams(asset_id="ASSET001")
        result = delete_asset(self.config, self.auth_manager, params)

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Asset deleted successfully")
        self.assertEqual(result.asset_id, "asset_sys_id")

    @patch("servicenow_mcp.tools.asset_tools._resolve_asset_id")
    def test_delete_asset_not_found(self, mock_resolve_asset):
        """Test asset deletion when asset is not found."""
        mock_resolve_asset.return_value = None

        params = DeleteAssetParams(asset_id="INVALID_ASSET")
        result = delete_asset(self.config, self.auth_manager, params)

        self.assertFalse(result.success)
        self.assertIn("Could not find asset", result.message)

    @patch("servicenow_mcp.tools.asset_tools._resolve_user_id")
    @patch("servicenow_mcp.tools.asset_tools._resolve_asset_id")
    @patch("servicenow_mcp.tools.asset_tools.requests.patch")
    def test_transfer_asset_success(self, mock_patch, mock_resolve_asset, mock_resolve_user):
        """Test successful asset transfer."""
        mock_resolve_asset.return_value = "asset_sys_id"
        mock_resolve_user.return_value = "new_user_sys_id"
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": {
                "sys_id": "asset_sys_id",
                "asset_tag": "ASSET001",
            }
        }
        mock_patch.return_value = mock_response

        params = TransferAssetParams(
            asset_id="ASSET001",
            new_assigned_to="jane.doe",
            transfer_reason="Employee transfer",
            comments="Transferring to new department",
        )

        result = transfer_asset(self.config, self.auth_manager, params)

        self.assertTrue(result.success)
        self.assertIn("transferred successfully", result.message)
        
        # Verify both resolutions were called
        mock_resolve_asset.assert_called_once_with(self.config, self.auth_manager, "ASSET001")
        mock_resolve_user.assert_called_once_with(self.config, self.auth_manager, "jane.doe")
        
        # Verify patch call includes transfer information
        call_args = mock_patch.call_args
        patch_data = call_args[1]["json"]
        self.assertEqual(patch_data["assigned_to"], "new_user_sys_id")
        self.assertIn("Asset transferred to jane.doe", patch_data["comments"])
        self.assertIn("Employee transfer", patch_data["comments"])

    @patch("servicenow_mcp.tools.asset_tools._resolve_asset_id")
    def test_transfer_asset_asset_not_found(self, mock_resolve_asset):
        """Test asset transfer when asset is not found."""
        mock_resolve_asset.return_value = None

        params = TransferAssetParams(
            asset_id="INVALID_ASSET",
            new_assigned_to="jane.doe",
        )

        result = transfer_asset(self.config, self.auth_manager, params)

        self.assertFalse(result.success)
        self.assertIn("Could not find asset", result.message)

    @patch("servicenow_mcp.tools.asset_tools._resolve_user_id")
    @patch("servicenow_mcp.tools.asset_tools._resolve_asset_id")
    def test_transfer_asset_user_not_found(self, mock_resolve_asset, mock_resolve_user):
        """Test asset transfer when new user is not found."""
        mock_resolve_asset.return_value = "asset_sys_id"
        mock_resolve_user.return_value = None

        params = TransferAssetParams(
            asset_id="ASSET001",
            new_assigned_to="invalid_user",
        )

        result = transfer_asset(self.config, self.auth_manager, params)

        self.assertFalse(result.success)
        self.assertIn("Could not resolve user", result.message)

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_resolve_user_id_by_username(self, mock_get):
        """Test user ID resolution by username."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [{"sys_id": "user_sys_id"}]
        }
        mock_get.return_value = mock_response

        result = _resolve_user_id(self.config, self.auth_manager, "john.doe")

        self.assertEqual(result, "user_sys_id")
        # Verify it tried username first
        call_args = mock_get.call_args
        self.assertIn("user_name=john.doe", call_args[1]["params"]["sysparm_query"])

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_resolve_user_id_by_email_fallback(self, mock_get):
        """Test user ID resolution falls back to email when username fails."""
        # Mock two calls - first for username (empty), second for email (success)
        mock_response_empty = Mock()
        mock_response_empty.raise_for_status.return_value = None
        mock_response_empty.json.return_value = {"result": []}
        
        mock_response_success = Mock()
        mock_response_success.raise_for_status.return_value = None
        mock_response_success.json.return_value = {
            "result": [{"sys_id": "user_sys_id"}]
        }
        
        mock_get.side_effect = [mock_response_empty, mock_response_success]

        result = _resolve_user_id(self.config, self.auth_manager, "john.doe@company.com")

        self.assertEqual(result, "user_sys_id")
        # Verify it made two calls
        self.assertEqual(mock_get.call_count, 2)

    def test_resolve_user_id_sys_id_passthrough(self):
        """Test user ID resolution passes through sys_id unchanged."""
        sys_id = "a1b2c3d4e5f6789012345678901234567"
        result = _resolve_user_id(self.config, self.auth_manager, sys_id)
        self.assertEqual(result, sys_id)

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_resolve_asset_id_by_tag(self, mock_get):
        """Test asset ID resolution by asset tag."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [{"sys_id": "asset_sys_id"}]
        }
        mock_get.return_value = mock_response

        result = _resolve_asset_id(self.config, self.auth_manager, "ASSET001")

        self.assertEqual(result, "asset_sys_id")
        call_args = mock_get.call_args
        self.assertIn("asset_tag=ASSET001", call_args[1]["params"]["sysparm_query"])

    def test_resolve_asset_id_sys_id_passthrough(self):
        """Test asset ID resolution passes through sys_id unchanged."""
        sys_id = "a1b2c3d4e5f6789012345678901234567"
        result = _resolve_asset_id(self.config, self.auth_manager, sys_id)
        self.assertEqual(result, sys_id)

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_resolve_asset_id_not_found(self, mock_get):
        """Test asset ID resolution when asset is not found."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": []}
        mock_get.return_value = mock_response

        result = _resolve_asset_id(self.config, self.auth_manager, "INVALID_TAG")

        self.assertIsNone(result)

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_success(self, mock_get):
        """Test successful hardware assets listing."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "hw_asset1",
                    "asset_tag": "HW001",
                    "display_name": "Dell Server",
                    "model": "Dell PowerEdge",
                    "serial_number": "SN001",
                },
                {
                    "sys_id": "hw_asset2",
                    "asset_tag": "HW002", 
                    "display_name": "HP Laptop",
                    "model": "HP EliteBook",
                    "serial_number": "SN002",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(limit=10)
        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["message"], "Found 2 hardware assets")
        self.assertIn("hardware_assets", result)
        self.assertEqual(len(result["hardware_assets"]), 2)

        # Verify API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertEqual(call_args[0][0], f"{self.config.api_url}/table/alm_hardware")
        self.assertIn("sysparm_limit", call_args[1]["params"])
        self.assertEqual(call_args[1]["params"]["sysparm_limit"], "10")
        self.assertEqual(call_args[1]["params"]["sysparm_display_value"], "true")

    @patch("servicenow_mcp.tools.asset_tools._resolve_user_id")
    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_with_assigned_to_filter(self, mock_get, mock_resolve_user):
        """Test hardware assets listing with assigned_to filter."""
        mock_resolve_user.return_value = "user_sys_id_123"
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "hw_asset1",
                    "asset_tag": "HW001",
                    "display_name": "Dell Server",
                    "assigned_to": "John Doe",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(
            limit=20,
            assigned_to="john.doe"
        )

        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        
        # Verify user resolution was called
        mock_resolve_user.assert_called_once_with(self.config, self.auth_manager, "john.doe")
        
        # Verify query includes assigned_to filter
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertEqual(query_param, "assigned_to=user_sys_id_123")

    @patch("servicenow_mcp.tools.asset_tools._resolve_user_id")
    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_user_not_resolved(self, mock_get, mock_resolve_user):
        """Test hardware assets listing when assigned user cannot be resolved."""
        mock_resolve_user.return_value = None
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "hw_asset1",
                    "asset_tag": "HW001",
                    "display_name": "Dell Server",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(
            assigned_to="invalid.user"
        )

        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        
        # Verify it falls back to direct match when user resolution fails
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertEqual(query_param, "assigned_to=invalid.user")

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_with_name_filter(self, mock_get):
        """Test hardware assets listing with name filter."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "hw_asset1",
                    "asset_tag": "HW001",
                    "display_name": "Dell PowerEdge Server",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(
            name="Dell PowerEdge",
            limit=15
        )

        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        
        # Verify the query includes name filter
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertEqual(query_param, "display_nameLIKEDell PowerEdge")
        self.assertEqual(call_args[1]["params"]["sysparm_limit"], "15")

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_with_query_filter(self, mock_get):
        """Test hardware assets listing with general query filter."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "hw_asset1",
                    "asset_tag": "LAPTOP001",
                    "display_name": "HP EliteBook",
                    "serial_number": "SN12345",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(
            query="LAPTOP",
            limit=25
        )

        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        
        # Verify the query includes general search filter
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertIn("asset_tagLIKELAPTOP", query_param)
        self.assertIn("display_nameLIKELAPTOP", query_param)
        self.assertIn("serial_numberLIKELAPTOP", query_param)
        self.assertIn("modelLIKELAPTOP", query_param)

    @patch("servicenow_mcp.tools.asset_tools._resolve_user_id")
    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_with_multiple_filters(self, mock_get, mock_resolve_user):
        """Test hardware assets listing with multiple filters."""
        mock_resolve_user.return_value = "user_sys_id_456"
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "result": [
                {
                    "sys_id": "hw_asset1",
                    "asset_tag": "SERVER001",
                    "display_name": "Dell PowerEdge Server",
                    "assigned_to": "Jane Doe",
                }
            ]
        }
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(
            assigned_to="jane.doe",
            name="Dell",
            query="SERVER",
            limit=5,
            offset=10
        )

        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)
        
        # Verify all filters are combined properly
        call_args = mock_get.call_args
        query_param = call_args[1]["params"]["sysparm_query"]
        self.assertIn("assigned_to=user_sys_id_456", query_param)
        self.assertIn("display_nameLIKEDell", query_param)
        self.assertIn("asset_tagLIKESERVER", query_param)
        
        # Verify pagination parameters
        self.assertEqual(call_args[1]["params"]["sysparm_limit"], "5")
        self.assertEqual(call_args[1]["params"]["sysparm_offset"], "10")

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_no_results(self, mock_get):
        """Test hardware assets listing with no results."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"result": []}
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(
            name="Nonexistent Hardware",
        )

        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["message"], "Found 0 hardware assets")
        self.assertEqual(result["hardware_assets"], [])

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_api_error(self, mock_get):
        """Test hardware assets listing with API error."""
        mock_get.side_effect = requests.RequestException("Connection timeout")

        params = ListHardwareAssetsParams(limit=10)
        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertFalse(result["success"])
        self.assertIn("Failed to list hardware assets", result["message"])
        self.assertIn("Connection timeout", result["message"])

    @patch("servicenow_mcp.tools.asset_tools.requests.get")
    def test_list_hardware_assets_http_error(self, mock_get):
        """Test hardware assets listing with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_get.return_value = mock_response

        params = ListHardwareAssetsParams(limit=10)
        result = list_hardware_assets(self.config, self.auth_manager, params)

        self.assertFalse(result["success"])
        self.assertIn("Failed to list hardware assets", result["message"])

    def test_list_hardware_assets_params_validation(self):
        """Test ListHardwareAssetsParams validation."""
        # Test valid parameters
        params = ListHardwareAssetsParams(
            limit=20,
            offset=5,
            assigned_to="john.doe",
            name="Dell Server",
            query="SERVER"
        )
        self.assertEqual(params.limit, 20)
        self.assertEqual(params.offset, 5)
        self.assertEqual(params.assigned_to, "john.doe")
        self.assertEqual(params.name, "Dell Server")
        self.assertEqual(params.query, "SERVER")
        
        # Test default values
        params_defaults = ListHardwareAssetsParams()
        self.assertEqual(params_defaults.limit, 10)
        self.assertEqual(params_defaults.offset, 0)
        self.assertIsNone(params_defaults.assigned_to)
        self.assertIsNone(params_defaults.name)
        self.assertIsNone(params_defaults.query)


if __name__ == "__main__":
    unittest.main()
