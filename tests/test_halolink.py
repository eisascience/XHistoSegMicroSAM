"""
Tests for Halo Link integration client
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from xhalo.halolink import HaloLinkClient


class TestHaloLinkClientInit:
    """Test HaloLinkClient initialization"""
    
    def test_client_init_with_base_url(self):
        """Test client initialization with base URL"""
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        assert client.base_url == "https://halolink.example.com"
        assert client._token is None
        assert client._token_expiry is None
    
    def test_client_init_from_env(self):
        """Test client initialization from environment variables"""
        with patch.dict(os.environ, {
            "HALOLINK_BASE_URL": "https://halolink.env.com",
            "HALOLINK_CLIENT_ID": "test-client",
            "HALOLINK_CLIENT_SECRET": "test-secret"
        }):
            client = HaloLinkClient()
            
            assert client.base_url == "https://halolink.env.com"
            assert client.client_id == "test-client"
            assert client.client_secret == "test-secret"
    
    def test_client_init_without_base_url_fails(self):
        """Test that initialization fails without base URL"""
        with patch.dict(os.environ, clear=True):
            with pytest.raises(ValueError, match="HALOLINK_BASE_URL is required"):
                HaloLinkClient()
    
    def test_client_strips_trailing_slash(self):
        """Test that trailing slashes are removed from base URL"""
        client = HaloLinkClient(base_url="https://halolink.example.com/")
        assert client.base_url == "https://halolink.example.com"


class TestOIDCDiscovery:
    """Test OIDC discovery functionality"""
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    def test_oidc_discovery_success(self, mock_get):
        """Test successful OIDC discovery"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "token_endpoint": "https://halolink.example.com/oauth/token",
            "issuer": "https://halolink.example.com"
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = HaloLinkClient(base_url="https://halolink.example.com")
        config = client._get_oidc_configuration()
        
        assert config["token_endpoint"] == "https://halolink.example.com/oauth/token"
        mock_get.assert_called_once_with(
            "https://halolink.example.com/.well-known/openid-configuration",
            timeout=10
        )
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    def test_oidc_discovery_connection_error(self, mock_get):
        """Test OIDC discovery with connection error (off VPN)"""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        with pytest.raises(RuntimeError, match="Are you on VPN"):
            client._get_oidc_configuration()
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    def test_oidc_discovery_timeout(self, mock_get):
        """Test OIDC discovery with timeout"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")
        
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        with pytest.raises(RuntimeError, match="Timeout"):
            client._get_oidc_configuration()
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    def test_oidc_discovery_http_error(self, mock_get):
        """Test OIDC discovery with HTTP error"""
        import requests
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = mock_response
        
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        with pytest.raises(RuntimeError, match="HTTP error"):
            client._get_oidc_configuration()


class TestTokenRetrieval:
    """Test OAuth2 token retrieval"""
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_token_retrieval_with_basic_auth(self, mock_post, mock_get):
        """Test token retrieval with HTTP basic auth"""
        # Mock OIDC discovery
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "token_endpoint": "https://halolink.example.com/oauth/token"
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response
        
        # Mock token response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "access_token": "test-token-123",
            "expires_in": 3600
        }
        mock_post_response.status_code = 200
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response
        
        client = HaloLinkClient(
            base_url="https://halolink.example.com",
            client_id="test-client",
            client_secret="test-secret"
        )
        
        token = client._retrieve_token()
        
        assert token == "test-token-123"
        assert client._token_expiry is not None
        mock_post.assert_called_once()
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_token_retrieval_fallback_to_form_fields(self, mock_post, mock_get):
        """Test token retrieval fallback to form fields when basic auth fails"""
        # Mock OIDC discovery
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "token_endpoint": "https://halolink.example.com/oauth/token"
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response
        
        # First call with basic auth fails with 401
        mock_401_response = Mock()
        mock_401_response.status_code = 401
        
        # Second call with form fields succeeds
        mock_200_response = Mock()
        mock_200_response.json.return_value = {
            "access_token": "test-token-456",
            "expires_in": 7200
        }
        mock_200_response.status_code = 200
        mock_200_response.raise_for_status = Mock()
        
        mock_post.side_effect = [mock_401_response, mock_200_response]
        
        client = HaloLinkClient(
            base_url="https://halolink.example.com",
            client_id="test-client",
            client_secret="test-secret"
        )
        
        token = client._retrieve_token()
        
        assert token == "test-token-456"
        assert mock_post.call_count == 2
    
    def test_token_retrieval_without_credentials_fails(self):
        """Test that token retrieval fails without credentials"""
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        with pytest.raises(ValueError, match="HALOLINK_CLIENT_ID and HALOLINK_CLIENT_SECRET are required"):
            client._retrieve_token()
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_get_token_caching(self, mock_post, mock_get):
        """Test that tokens are cached and reused"""
        # Mock OIDC discovery
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "token_endpoint": "https://halolink.example.com/oauth/token"
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response
        
        # Mock token response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "access_token": "cached-token",
            "expires_in": 3600
        }
        mock_post_response.status_code = 200
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response
        
        client = HaloLinkClient(
            base_url="https://halolink.example.com",
            client_id="test-client",
            client_secret="test-secret"
        )
        
        # First call should retrieve token
        token1 = client.get_token()
        
        # Second call should use cached token
        token2 = client.get_token()
        
        assert token1 == token2 == "cached-token"
        assert mock_post.call_count == 1  # Only called once
    
    def test_get_token_without_credentials_returns_none(self):
        """Test that get_token returns None without credentials"""
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        token = client.get_token()
        
        assert token is None


class TestGraphQLExecution:
    """Test GraphQL query execution"""
    
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_execute_graphql_success(self, mock_post):
        """Test successful GraphQL query execution"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "__typename": "Query"
            }
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        result = client.execute_graphql("{ __typename }")
        
        assert result["data"]["__typename"] == "Query"
        mock_post.assert_called_once()
    
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_execute_graphql_with_variables(self, mock_post):
        """Test GraphQL query with variables"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "user": {"id": "123", "name": "Test User"}
            }
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        query = "query GetUser($id: ID!) { user(id: $id) { id name } }"
        variables = {"id": "123"}
        
        result = client.execute_graphql(query, variables)
        
        assert result["data"]["user"]["id"] == "123"
        
        # Verify variables were sent
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["variables"] == variables
    
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_execute_graphql_with_token(self, mock_post):
        """Test GraphQL query with Bearer token"""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {}}
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = HaloLinkClient(
            base_url="https://halolink.example.com",
            client_id="test-client",
            client_secret="test-secret"
        )
        
        # Mock cached token
        client._token = "test-token-789"
        client._token_expiry = datetime.now() + timedelta(hours=1)
        
        client.execute_graphql("{ __typename }")
        
        # Verify Bearer token was sent
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-token-789"
    
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_execute_graphql_errors(self, mock_post):
        """Test GraphQL query with errors"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "errors": [
                {"message": "Field not found"}
            ]
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        with pytest.raises(RuntimeError, match="GraphQL errors"):
            client.execute_graphql("{ invalidField }")
    
    def test_get_graphql_endpoint_with_direct_url(self):
        """Test GraphQL endpoint resolution with direct URL"""
        client = HaloLinkClient(
            base_url="https://halolink.example.com",
            graphql_url="https://halolink.example.com/api/v2/graphql"
        )
        
        endpoint = client._get_graphql_endpoint()
        
        assert endpoint == "https://halolink.example.com/api/v2/graphql"
    
    def test_get_graphql_endpoint_with_path(self):
        """Test GraphQL endpoint resolution with path"""
        client = HaloLinkClient(
            base_url="https://halolink.example.com",
            graphql_path="/api/graphql"
        )
        
        endpoint = client._get_graphql_endpoint()
        
        assert endpoint == "https://halolink.example.com/api/graphql"
    
    def test_get_graphql_endpoint_with_path_no_leading_slash(self):
        """Test GraphQL endpoint resolution with path without leading slash"""
        client = HaloLinkClient(
            base_url="https://halolink.example.com",
            graphql_path="api/graphql"
        )
        
        endpoint = client._get_graphql_endpoint()
        
        # Should normalize path to include leading slash
        assert endpoint == "https://halolink.example.com/api/graphql"
    
    def test_get_graphql_endpoint_default(self):
        """Test GraphQL endpoint resolution with default"""
        client = HaloLinkClient(base_url="https://halolink.example.com")
        
        endpoint = client._get_graphql_endpoint()
        
        assert endpoint == "https://halolink.example.com/graphql"


class TestSmokeTest:
    """Test smoke test functionality"""
    
    @patch('xhalo.halolink.halolink_client.requests.get')
    @patch('xhalo.halolink.halolink_client.requests.post')
    def test_smoke_test_without_auth(self, mock_post, mock_get):
        """Test smoke test without authentication"""
        from xhalo.halolink.smoketest import run_smoke_test
        
        # Mock OIDC discovery
        mock_get_response = Mock()
        mock_get_response.json.return_value = {
            "token_endpoint": "https://halolink.example.com/oauth/token"
        }
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response
        
        # Mock GraphQL response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {
            "data": {"__typename": "Query"}
        }
        mock_post_response.status_code = 200
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response
        
        with patch.dict(os.environ, {"HALOLINK_BASE_URL": "https://halolink.example.com"}):
            results = run_smoke_test(verbose=False)
        
        assert results["success"] is True
        assert results["steps"]["client_init"]["success"] is True
        assert results["steps"]["oidc_discovery"]["success"] is True
        assert results["steps"]["graphql_query"]["success"] is True
