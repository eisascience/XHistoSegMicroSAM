"""
Halo Link Client

Implements OIDC discovery, OAuth2 client_credentials authentication,
and GraphQL request capabilities for Halo Link platform.
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)


class HaloLinkClient:
    """
    Client for Halo Link integration with OIDC discovery and OAuth2 authentication.
    
    Environment Variables:
        HALOLINK_BASE_URL: Base URL for Halo Link (required)
        HALOLINK_GRAPHQL_URL: GraphQL endpoint URL (optional, preferred)
        HALOLINK_GRAPHQL_PATH: GraphQL path (optional, used if GRAPHQL_URL not set)
        HALOLINK_CLIENT_ID: OAuth2 client ID (optional, for auth)
        HALOLINK_CLIENT_SECRET: OAuth2 client secret (optional, for auth)
        HALOLINK_SCOPE: OAuth2 scope (optional)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        graphql_url: Optional[str] = None,
        graphql_path: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scope: Optional[str] = None,
    ):
        """
        Initialize Halo Link client.
        
        Args:
            base_url: Base URL for Halo Link (defaults to HALOLINK_BASE_URL env var)
            graphql_url: GraphQL endpoint URL (defaults to HALOLINK_GRAPHQL_URL env var)
            graphql_path: GraphQL path (defaults to HALOLINK_GRAPHQL_PATH env var)
            client_id: OAuth2 client ID (defaults to HALOLINK_CLIENT_ID env var)
            client_secret: OAuth2 client secret (defaults to HALOLINK_CLIENT_SECRET env var)
            scope: OAuth2 scope (defaults to HALOLINK_SCOPE env var)
        """
        # Load configuration from environment if not provided
        self.base_url = base_url or os.getenv("HALOLINK_BASE_URL")
        self.graphql_url = graphql_url or os.getenv("HALOLINK_GRAPHQL_URL")
        self.graphql_path = graphql_path or os.getenv("HALOLINK_GRAPHQL_PATH")
        self.client_id = client_id or os.getenv("HALOLINK_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("HALOLINK_CLIENT_SECRET")
        self.scope = scope or os.getenv("HALOLINK_SCOPE")
        
        # Validate required configuration
        if not self.base_url:
            raise ValueError(
                "HALOLINK_BASE_URL is required. Please set it in your environment "
                "or .env file. Example: HALOLINK_BASE_URL=https://halolink.example.com"
            )
        
        # Remove trailing slash from base_url
        self.base_url = self.base_url.rstrip("/")
        
        # Token cache
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._token_endpoint: Optional[str] = None
        
        logger.info(f"HaloLinkClient initialized with base_url: {self.base_url}")
    
    def _get_oidc_configuration(self) -> Dict[str, Any]:
        """
        Perform OIDC discovery to get token endpoint.
        
        Returns:
            Dict containing OIDC configuration
            
        Raises:
            RuntimeError: If discovery fails (likely off VPN or wrong base URL)
        """
        discovery_url = f"{self.base_url}/.well-known/openid-configuration"
        
        logger.info(f"Attempting OIDC discovery at: {discovery_url}")
        
        try:
            response = requests.get(discovery_url, timeout=10)
            response.raise_for_status()
            
            config = response.json()
            logger.info("OIDC discovery successful")
            logger.debug(f"OIDC configuration: {config}")
            
            return config
            
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Failed to connect to {discovery_url}. "
                f"Are you on VPN? Is the base URL correct? Error: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                f"Timeout connecting to {discovery_url}. "
                f"Are you on VPN? Error: {e}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"HTTP error during OIDC discovery: {e}. "
                f"Status code: {response.status_code}. "
                f"Check if the base URL is correct."
            )
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during OIDC discovery: {e}. "
                f"Check your HALOLINK_BASE_URL configuration."
            )
    
    def _get_token_endpoint(self) -> str:
        """
        Get token endpoint from OIDC discovery.
        
        Returns:
            Token endpoint URL
        """
        if not self._token_endpoint:
            config = self._get_oidc_configuration()
            self._token_endpoint = config.get("token_endpoint")
            
            if not self._token_endpoint:
                raise RuntimeError(
                    "token_endpoint not found in OIDC configuration. "
                    "The server may not support OAuth2."
                )
            
            logger.info(f"Token endpoint: {self._token_endpoint}")
        
        return self._token_endpoint
    
    def _retrieve_token(self) -> str:
        """
        Retrieve OAuth2 token using client_credentials grant.
        
        Returns:
            Access token string
            
        Raises:
            RuntimeError: If token retrieval fails
            ValueError: If credentials are not configured
        """
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "HALOLINK_CLIENT_ID and HALOLINK_CLIENT_SECRET are required for authentication. "
                "Please set them in your environment or .env file."
            )
        
        token_endpoint = self._get_token_endpoint()
        
        # Prepare token request data
        data = {
            "grant_type": "client_credentials",
        }
        
        if self.scope:
            data["scope"] = self.scope
        
        # Try HTTP basic auth first (recommended by OAuth2 spec)
        logger.info(f"Requesting token from: {token_endpoint}")
        logger.debug(f"Using client_id: {self.client_id}")
        
        try:
            # Try with HTTP basic auth
            response = requests.post(
                token_endpoint,
                data=data,
                auth=(self.client_id, self.client_secret),
                timeout=10
            )
            
            # If basic auth fails with 401, try with form fields
            if response.status_code == 401:
                logger.debug("Basic auth failed, trying form fields")
                data["client_id"] = self.client_id
                data["client_secret"] = self.client_secret
                
                response = requests.post(
                    token_endpoint,
                    data=data,
                    timeout=10
                )
            
            response.raise_for_status()
            
            token_data = response.json()
            access_token = token_data.get("access_token")
            
            if not access_token:
                raise RuntimeError("access_token not found in token response")
            
            # Calculate token expiry (default to 1 hour if not provided)
            expires_in = token_data.get("expires_in", 3600)
            self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info(f"Token retrieved successfully (expires in {expires_in}s)")
            logger.debug(f"Token expiry: {self._token_expiry}")
            
            return access_token
            
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Failed to retrieve token. HTTP {response.status_code}: {response.text}. "
                f"Check your client credentials."
            )
        except Exception as e:
            raise RuntimeError(f"Error retrieving token: {e}")
    
    def get_token(self) -> Optional[str]:
        """
        Get cached token or retrieve new one if expired.
        
        Returns:
            Access token string, or None if credentials not configured
        """
        # If no credentials, return None (anonymous access)
        if not self.client_id or not self.client_secret:
            logger.debug("No credentials configured, skipping authentication")
            return None
        
        # Check if token is cached and still valid
        if self._token and self._token_expiry:
            # Refresh if token expires in less than 5 minutes
            if datetime.now() < (self._token_expiry - timedelta(minutes=5)):
                logger.debug("Using cached token")
                return self._token
        
        # Retrieve new token
        logger.info("Token expired or not cached, retrieving new token")
        self._token = self._retrieve_token()
        return self._token
    
    def _get_graphql_endpoint(self) -> str:
        """
        Get GraphQL endpoint URL.
        
        Returns:
            GraphQL endpoint URL
        """
        if self.graphql_url:
            return self.graphql_url
        elif self.graphql_path:
            # Normalize path to ensure it starts with /
            path = self.graphql_path if self.graphql_path.startswith('/') else f'/{self.graphql_path}'
            return f"{self.base_url}{path}"
        else:
            # Default to /graphql if not specified
            return f"{self.base_url}/graphql"
    
    def execute_graphql(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query.
        
        Args:
            query: GraphQL query string
            variables: Optional query variables
            
        Returns:
            GraphQL response data
            
        Raises:
            RuntimeError: If request fails
        """
        endpoint = self._get_graphql_endpoint()
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add authorization if token available
        token = self.get_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
            logger.debug("Using Bearer token authentication")
        else:
            logger.debug("No authentication token available")
        
        payload = {
            "query": query,
        }
        
        if variables:
            payload["variables"] = variables
        
        # Log request (without secrets)
        logger.info(f"Executing GraphQL request to: {endpoint}")
        logger.debug(f"Query: {query}")
        if variables:
            logger.debug(f"Variables: {variables}")
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            logger.info(f"GraphQL response status: {response.status_code}")
            
            response.raise_for_status()
            
            result = response.json()
            
            # Check for GraphQL errors
            if "errors" in result:
                logger.error(f"GraphQL errors: {result['errors']}")
                raise RuntimeError(f"GraphQL errors: {result['errors']}")
            
            logger.debug(f"GraphQL response: {result}")
            
            return result
            
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Failed to connect to {endpoint}. "
                f"Are you on VPN? Error: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Timeout executing GraphQL request: {e}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"HTTP error executing GraphQL request: {e}. "
                f"Status code: {response.status_code}. "
                f"Response: {response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Error executing GraphQL request: {e}")
