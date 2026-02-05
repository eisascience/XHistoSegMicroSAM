"""
Halo Digital Pathology API Integration

Provides interface to Halo's GraphQL API for:
- Fetching slides and metadata
- Downloading image regions
- Querying annotations
- Uploading analysis results
"""

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
from gql.transport.exceptions import TransportQueryError
import requests
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HaloAPI:
    """
    Interface for Halo Digital Pathology GraphQL API.
    """
    
    def __init__(self, endpoint: str, token: str):
        """
        Initialize Halo API client.
        
        Args:
            endpoint: GraphQL endpoint URL (e.g., https://halo.example.com/graphql)
            token: API authentication token
        """
        self.endpoint = endpoint
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Setup GraphQL client with authentication
        transport = AIOHTTPTransport(
            url=endpoint,
            headers=self.headers
        )
        
        try:
            self.client = Client(
                transport=transport,
                fetch_schema_from_transport=True
            )
            logger.info(f"Initialized Halo API client for {endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize API client: {str(e)}")
            raise
    
    async def test_connection(self) -> bool:
        """
        Test API connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            await self.get_slides(limit=1)
            logger.info("API connection test successful")
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {str(e)}")
            return False
    
    async def get_slides(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch list of available slides.
        
        Args:
            limit: Maximum number of slides to return
            offset: Number of slides to skip (for pagination)
            
        Returns:
            List of slide dictionaries with metadata
        """
        query = gql('''
            query GetSlides($limit: Int!, $offset: Int!) {
                slides(first: $limit, offset: $offset) {
                    edges {
                        node {
                            id
                            name
                            width
                            height
                            mpp
                            tileSize
                            format
                            studyId
                            createdAt
                            updatedAt
                        }
                    }
                }
            }
        ''')
        
        try:
            result = await self.client.execute_async(
                query,
                variable_values={"limit": limit, "offset": offset}
            )
            slides = [edge['node'] for edge in result['slides']['edges']]
            logger.info(f"Retrieved {len(slides)} slides")
            return slides
        except TransportQueryError as e:
            logger.error(f"GraphQL query error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch slides: {str(e)}")
            raise
    
    async def get_slide_by_id(self, slide_id: str) -> Dict:
        """
        Fetch detailed metadata for a specific slide.
        
        Args:
            slide_id: Unique slide identifier
            
        Returns:
            Dictionary with slide metadata
        """
        query = gql('''
            query GetSlide($id: ID!) {
                slide(id: $id) {
                    id
                    name
                    width
                    height
                    mpp
                    tileSize
                    format
                    metadata
                    studyId
                    createdAt
                    updatedAt
                }
            }
        ''')
        
        try:
            result = await self.client.execute_async(
                query,
                variable_values={"id": slide_id}
            )
            logger.info(f"Retrieved metadata for slide {slide_id}")
            return result['slide']
        except Exception as e:
            logger.error(f"Failed to fetch slide metadata: {str(e)}")
            raise
    
    def download_region(self, slide_id: str, x: int, y: int, 
                       width: int, height: int, level: int = 0) -> bytes:
        """
        Download a specific rectangular region from a slide.
        
        Args:
            slide_id: Unique slide identifier
            x: Left coordinate (pixels)
            y: Top coordinate (pixels)
            width: Region width (pixels)
            height: Region height (pixels)
            level: Pyramid level (0 = highest resolution)
            
        Returns:
            Image data as bytes
        """
        # Construct region download URL
        base_url = self.endpoint.replace('/graphql', '')
        url = f"{base_url}/api/slides/{slide_id}/region"
        
        params = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "level": level,
            "format": "jpeg"
        }
        
        try:
            logger.info(f"Downloading region: {x},{y} {width}x{height} from slide {slide_id}")
            response = requests.get(url, params=params, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            logger.info(f"Downloaded {len(response.content)} bytes")
            return response.content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download region: {str(e)}")
            raise
    
    def download_thumbnail(self, slide_id: str, max_size: int = 512) -> bytes:
        """
        Download slide thumbnail.
        
        Args:
            slide_id: Unique slide identifier
            max_size: Maximum dimension (pixels)
            
        Returns:
            Thumbnail image data as bytes
        """
        base_url = self.endpoint.replace('/graphql', '')
        url = f"{base_url}/api/slides/{slide_id}/thumbnail"
        
        params = {"max_size": max_size}
        
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download thumbnail: {str(e)}")
            raise
    
    async def get_annotations(self, slide_id: str) -> List[Dict]:
        """
        Fetch annotations for a slide.
        
        Args:
            slide_id: Unique slide identifier
            
        Returns:
            List of annotation dictionaries
        """
        query = gql('''
            query GetAnnotations($slideId: ID!) {
                annotations(slideId: $slideId) {
                    edges {
                        node {
                            id
                            name
                            type
                            geometry
                            properties
                            createdAt
                        }
                    }
                }
            }
        ''')
        
        try:
            result = await self.client.execute_async(
                query,
                variable_values={"slideId": slide_id}
            )
            annotations = [edge['node'] for edge in result['annotations']['edges']]
            logger.info(f"Retrieved {len(annotations)} annotations")
            return annotations
        except Exception as e:
            logger.error(f"Failed to fetch annotations: {str(e)}")
            raise
    
    async def upload_annotations(self, slide_id: str, geojson: Dict, 
                                 name: str = "MedSAM Analysis") -> Dict:
        """
        Upload annotations to Halo.
        
        Args:
            slide_id: Target slide identifier
            geojson: GeoJSON FeatureCollection
            name: Annotation layer name
            
        Returns:
            Created annotation metadata
        """
        mutation = gql('''
            mutation CreateAnnotation($slideId: ID!, $name: String!, $geojson: JSON!) {
                createAnnotation(input: {
                    slideId: $slideId
                    name: $name
                    geometry: $geojson
                }) {
                    annotation {
                        id
                        name
                        createdAt
                    }
                }
            }
        ''')
        
        try:
            result = await self.client.execute_async(
                mutation,
                variable_values={
                    "slideId": slide_id,
                    "name": name,
                    "geojson": geojson
                }
            )
            logger.info(f"Successfully uploaded annotations to slide {slide_id}")
            return result['createAnnotation']['annotation']
        except Exception as e:
            logger.error(f"Failed to upload annotations: {str(e)}")
            raise
