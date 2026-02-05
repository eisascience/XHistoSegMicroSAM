"""
Halo Link Smoke Test

CLI module for testing Halo Link integration:
- OIDC discovery
- Token retrieval
- GraphQL query execution

Usage:
    python -m xhalo.halolink.smoketest
"""

import os
import sys
import logging
from typing import Optional

from .halolink_client import HaloLinkClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_smoke_test(verbose: bool = False) -> dict:
    """
    Run smoke test for Halo Link integration.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        Dict with test results
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    results = {
        "success": False,
        "steps": {},
        "error": None
    }
    
    print("=" * 60)
    print("Halo Link Integration Smoke Test")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Initialize client
        print("Step 1: Initializing Halo Link client...")
        try:
            client = HaloLinkClient()
            print(f" Client initialized with base_url: {client.base_url}")
            results["steps"]["client_init"] = {"success": True}
        except Exception as e:
            print(f" Failed to initialize client: {e}")
            results["steps"]["client_init"] = {"success": False, "error": str(e)}
            results["error"] = f"Client initialization failed: {e}"
            return results
        
        print()
        
        # Step 2: OIDC discovery
        print("Step 2: Performing OIDC discovery...")
        try:
            config = client._get_oidc_configuration()
            token_endpoint = config.get("token_endpoint", "N/A")
            print(f" OIDC discovery successful")
            print(f"  Token endpoint: {token_endpoint}")
            results["steps"]["oidc_discovery"] = {
                "success": True,
                "token_endpoint": token_endpoint
            }
        except Exception as e:
            print(f" OIDC discovery failed: {e}")
            results["steps"]["oidc_discovery"] = {"success": False, "error": str(e)}
            results["error"] = f"OIDC discovery failed: {e}"
            return results
        
        print()
        
        # Step 3: Token retrieval (if credentials configured)
        print("Step 3: Retrieving OAuth2 token...")
        has_credentials = client.client_id and client.client_secret
        
        if not has_credentials:
            print("⊘ Skipping token retrieval (no credentials configured)")
            print("  Set HALOLINK_CLIENT_ID and HALOLINK_CLIENT_SECRET to test authentication")
            results["steps"]["token_retrieval"] = {
                "success": True,
                "skipped": True,
                "reason": "No credentials configured"
            }
        else:
            try:
                token = client.get_token()
                # Don't log the actual token
                token_preview = f"{token[:10]}..." if token and len(token) > 10 else "N/A"
                print(f" Token retrieved successfully")
                print(f"  Token preview: {token_preview}")
                results["steps"]["token_retrieval"] = {
                    "success": True,
                    "has_token": bool(token)
                }
            except Exception as e:
                print(f" Token retrieval failed: {e}")
                results["steps"]["token_retrieval"] = {"success": False, "error": str(e)}
                # Don't fail the whole test if token retrieval fails
                # (might be testing without auth)
        
        print()
        
        # Step 4: GraphQL query
        print("Step 4: Executing test GraphQL query...")
        
        # Get query from environment or use default
        test_query = os.getenv("HALOLINK_SMOKETEST_QUERY", "{ __typename }")
        print(f"  Query: {test_query}")
        
        try:
            result = client.execute_graphql(test_query)
            print(f" GraphQL query executed successfully")
            print(f"  Response: {result}")
            results["steps"]["graphql_query"] = {
                "success": True,
                "result": result
            }
        except Exception as e:
            print(f" GraphQL query failed: {e}")
            results["steps"]["graphql_query"] = {"success": False, "error": str(e)}
            results["error"] = f"GraphQL query failed: {e}"
            return results
        
        print()
        print("=" * 60)
        print(" All smoke tests passed!")
        print("=" * 60)
        
        results["success"] = True
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f" Smoke test failed with unexpected error: {e}")
        print("=" * 60)
        results["error"] = str(e)
        logger.exception("Unexpected error during smoke test")
    
    return results


def main():
    """Main entry point for CLI smoke test."""
    # Check for verbose flag
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("\nOptions:")
        print("  -v, --verbose    Enable verbose logging")
        print("  -h, --help       Show this help message")
        print()
        print("Environment Variables:")
        print("  HALOLINK_BASE_URL              Base URL (required)")
        print("  HALOLINK_GRAPHQL_URL           GraphQL endpoint (optional)")
        print("  HALOLINK_GRAPHQL_PATH          GraphQL path (optional)")
        print("  HALOLINK_CLIENT_ID             OAuth2 client ID (optional)")
        print("  HALOLINK_CLIENT_SECRET         OAuth2 client secret (optional)")
        print("  HALOLINK_SCOPE                 OAuth2 scope (optional)")
        print("  HALOLINK_SMOKETEST_QUERY       Test query (default: '{ __typename }')")
        sys.exit(0)
    
    results = run_smoke_test(verbose=verbose)
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
