"""
Halo Link Integration Module

Provides OIDC discovery, OAuth2 authentication, and GraphQL integration
for Halo Link platform.
"""

from .halolink_client import HaloLinkClient

__all__ = ["HaloLinkClient"]
