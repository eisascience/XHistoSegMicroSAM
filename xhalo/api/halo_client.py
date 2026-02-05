"""
Halo GraphQL API Client Module
Provides integration with Halo's GraphQL API for exporting WSIs and ROIs
"""

from typing import Dict, List, Optional, Any
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import logging

logger = logging.getLogger(__name__)


class HaloAPIClient:
    """Client for interacting with Halo's GraphQL API"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        """
        Initialize Halo API client
        
        Args:
            api_url: URL of the Halo GraphQL API endpoint
            api_key: Optional API key for authentication
        """
        self.api_url = api_url
        self.api_key = api_key
        
        # Setup transport with authentication if provided
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        transport = AIOHTTPTransport(url=api_url, headers=headers)
        self.client = Client(transport=transport, fetch_schema_from_transport=True)
    
    async def list_slides(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available slides/WSIs from Halo
        
        Args:
            project_id: Optional project ID to filter slides
            
        Returns:
            List of slide metadata dictionaries
        """
        query = gql("""
            query ListSlides($projectId: ID) {
                slides(projectId: $projectId) {
                    id
                    name
                    path
                    width
                    height
                    magnification
                    metadata
                }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                query, 
                variable_values={"projectId": project_id} if project_id else {}
            )
            return result.get("slides", [])
        except Exception as e:
            logger.error(f"Error listing slides: {e}")
            return []
    
    async def get_slide_info(self, slide_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific slide
        
        Args:
            slide_id: ID of the slide
            
        Returns:
            Slide metadata dictionary or None if not found
        """
        query = gql("""
            query GetSlide($slideId: ID!) {
                slide(id: $slideId) {
                    id
                    name
                    path
                    width
                    height
                    magnification
                    pixelSizeX
                    pixelSizeY
                    metadata
                }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                query,
                variable_values={"slideId": slide_id}
            )
            return result.get("slide")
        except Exception as e:
            logger.error(f"Error getting slide info: {e}")
            return None
    
    async def export_roi(
        self, 
        slide_id: str, 
        roi_id: str,
        format: str = "image"
    ) -> Optional[bytes]:
        """
        Export a Region of Interest (ROI) from a slide
        
        Args:
            slide_id: ID of the slide
            roi_id: ID of the ROI to export
            format: Export format (image, geojson)
            
        Returns:
            Exported ROI data as bytes or None if failed
        """
        query = gql("""
            query ExportROI($slideId: ID!, $roiId: ID!, $format: String!) {
                exportROI(slideId: $slideId, roiId: $roiId, format: $format) {
                    data
                    contentType
                }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                query,
                variable_values={
                    "slideId": slide_id,
                    "roiId": roi_id,
                    "format": format
                }
            )
            return result.get("exportROI", {}).get("data")
        except Exception as e:
            logger.error(f"Error exporting ROI: {e}")
            return None
    
    async def list_rois(self, slide_id: str) -> List[Dict[str, Any]]:
        """
        List all ROIs for a given slide
        
        Args:
            slide_id: ID of the slide
            
        Returns:
            List of ROI metadata dictionaries
        """
        query = gql("""
            query ListROIs($slideId: ID!) {
                rois(slideId: $slideId) {
                    id
                    name
                    type
                    geometry
                    properties
                }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                query,
                variable_values={"slideId": slide_id}
            )
            return result.get("rois", [])
        except Exception as e:
            logger.error(f"Error listing ROIs: {e}")
            return []
    
    async def import_annotations(
        self,
        slide_id: str,
        annotations: List[Dict[str, Any]],
        layer_name: str = "AI Annotations"
    ) -> bool:
        """
        Import annotations/segmentations back to Halo
        
        Args:
            slide_id: ID of the slide to import annotations to
            annotations: List of annotation objects (GeoJSON format)
            layer_name: Name of the annotation layer
            
        Returns:
            True if successful, False otherwise
        """
        mutation = gql("""
            mutation ImportAnnotations(
                $slideId: ID!,
                $annotations: [AnnotationInput!]!,
                $layerName: String!
            ) {
                importAnnotations(
                    slideId: $slideId,
                    annotations: $annotations,
                    layerName: $layerName
                ) {
                    success
                    message
                }
            }
        """)
        
        try:
            result = await self.client.execute_async(
                mutation,
                variable_values={
                    "slideId": slide_id,
                    "annotations": annotations,
                    "layerName": layer_name
                }
            )
            return result.get("importAnnotations", {}).get("success", False)
        except Exception as e:
            logger.error(f"Error importing annotations: {e}")
            return False


class MockHaloAPIClient(HaloAPIClient):
    """Mock implementation for testing without actual Halo API"""
    
    def __init__(self, api_url: str = "http://mock-halo-api", api_key: Optional[str] = None):
        """Initialize mock client without actual connection"""
        self.api_url = api_url
        self.api_key = api_key
        self.client = None
        
        # Mock data
        self.mock_slides = [
            {
                "id": "slide_001",
                "name": "Sample Slide 1",
                "path": "/path/to/slide1.svs",
                "width": 50000,
                "height": 40000,
                "magnification": 40,
                "metadata": {}
            },
            {
                "id": "slide_002",
                "name": "Sample Slide 2",
                "path": "/path/to/slide2.svs",
                "width": 60000,
                "height": 45000,
                "magnification": 40,
                "metadata": {}
            }
        ]
    
    async def list_slides(self, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return mock slide list"""
        logger.info("Using mock Halo API - returning sample slides")
        return self.mock_slides
    
    async def get_slide_info(self, slide_id: str) -> Optional[Dict[str, Any]]:
        """Return mock slide info"""
        for slide in self.mock_slides:
            if slide["id"] == slide_id:
                return slide
        return None
    
    async def export_roi(
        self, 
        slide_id: str, 
        roi_id: str,
        format: str = "image"
    ) -> Optional[bytes]:
        """Return mock ROI export"""
        logger.info(f"Mock export ROI: slide={slide_id}, roi={roi_id}, format={format}")
        return b"mock_image_data"
    
    async def list_rois(self, slide_id: str) -> List[Dict[str, Any]]:
        """Return mock ROI list"""
        return [
            {
                "id": "roi_001",
                "name": "Tissue Region 1",
                "type": "polygon",
                "geometry": {},
                "properties": {}
            }
        ]
    
    async def import_annotations(
        self,
        slide_id: str,
        annotations: List[Dict[str, Any]],
        layer_name: str = "AI Annotations"
    ) -> bool:
        """Mock successful annotation import"""
        logger.info(f"Mock import {len(annotations)} annotations to slide {slide_id}")
        return True
