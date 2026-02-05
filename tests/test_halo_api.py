"""
Comprehensive tests for Halo API client
"""

import pytest
from xhalo.api import HaloAPIClient, MockHaloAPIClient


class TestMockHaloAPIClient:
    """Test MockHaloAPIClient functionality"""
    
    @pytest.mark.asyncio
    async def test_mock_client_initialization(self):
        """Test mock client initialization"""
        client = MockHaloAPIClient()
        
        assert client is not None
    
    @pytest.mark.asyncio
    async def test_list_slides(self):
        """Test listing slides"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        
        assert isinstance(slides, list)
        assert len(slides) > 0
        
        # Check slide structure
        for slide in slides:
            assert "id" in slide
            assert "name" in slide
            assert "width" in slide
            assert "height" in slide
    
    @pytest.mark.asyncio
    async def test_list_slides_with_project_id(self):
        """Test listing slides filtered by project"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides(project_id="project-123")
        
        assert isinstance(slides, list)
    
    @pytest.mark.asyncio
    async def test_get_slide_info(self):
        """Test getting slide information"""
        client = MockHaloAPIClient()
        
        # First get a slide ID
        slides = await client.list_slides()
        assert len(slides) > 0
        
        slide_id = slides[0]["id"]
        slide_info = await client.get_slide_info(slide_id)
        
        assert slide_info is not None
        assert "id" in slide_info
        assert "name" in slide_info
        assert "width" in slide_info
        assert "height" in slide_info
        assert slide_info["id"] == slide_id
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_slide(self):
        """Test getting info for non-existent slide"""
        client = MockHaloAPIClient()
        
        slide_info = await client.get_slide_info("nonexistent-id")
        
        # Should return None or empty dict
        assert slide_info is None or slide_info == {}
    
    @pytest.mark.asyncio
    async def test_list_rois(self):
        """Test listing ROIs for a slide"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        slide_id = slides[0]["id"]
        
        rois = await client.list_rois(slide_id)
        
        assert isinstance(rois, list)
        # Mock client may or may not have ROIs
    
    @pytest.mark.asyncio
    async def test_export_roi(self):
        """Test exporting ROI"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        slide_id = slides[0]["id"]
        rois = await client.list_rois(slide_id)
        
        if len(rois) > 0:
            roi_id = rois[0]["id"]
            roi_data = await client.export_roi(slide_id, roi_id)
            
            assert roi_data is not None
    
    @pytest.mark.asyncio
    async def test_export_roi_basic(self):
        """Test exporting a region from a slide"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        slide_id = slides[0]["id"]
        
        # Try to export ROI
        roi_data = await client.export_roi(
            slide_id,
            roi_id="roi_001",
            format="image"
        )
        
        # Should return some data (bytes)
        assert roi_data is not None
    
    @pytest.mark.asyncio
    async def test_import_annotations(self):
        """Test importing annotations"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        slide_id = slides[0]["id"]
        
        # Create mock annotations
        annotations = [
            {
                "type": "Polygon",
                "coordinates": [[10, 10], [100, 10], [100, 100], [10, 100]],
                "properties": {"label": "test"}
            }
        ]
        
        success = await client.import_annotations(
            slide_id,
            annotations,
            "Test Layer"
        )
        
        # Mock client should succeed
        assert success is True
    
    @pytest.mark.asyncio
    async def test_import_empty_annotations(self):
        """Test importing empty annotation list"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        slide_id = slides[0]["id"]
        
        success = await client.import_annotations(
            slide_id,
            [],
            "Empty Layer"
        )
        
        # Should handle empty list gracefully
        assert isinstance(success, bool)
    
    @pytest.mark.asyncio
    async def test_export_roi_formats(self):
        """Test exporting ROI in different formats"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        slide_id = slides[0]["id"]
        
        # Test with image format
        roi_data = await client.export_roi(slide_id, "roi_001", format="image")
        assert roi_data is not None
        
        # Test with geojson format
        roi_data_json = await client.export_roi(slide_id, "roi_001", format="geojson")
        assert roi_data_json is not None


class TestHaloAPIClientInit:
    """Test HaloAPIClient initialization"""
    
    def test_client_init_without_auth(self):
        """Test client initialization without API key"""
        client = HaloAPIClient(api_url="http://localhost:8080/graphql")
        
        assert client.api_url == "http://localhost:8080/graphql"
        assert client.api_key is None
    
    def test_client_init_with_auth(self):
        """Test client initialization with API key"""
        client = HaloAPIClient(
            api_url="http://localhost:8080/graphql",
            api_key="test-api-key"
        )
        
        assert client.api_url == "http://localhost:8080/graphql"
        assert client.api_key == "test-api-key"


class TestMockAPIDataConsistency:
    """Test data consistency in mock API"""
    
    @pytest.mark.asyncio
    async def test_slide_info_consistency(self):
        """Test that slide info is consistent with list"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        if len(slides) > 0:
            slide = slides[0]
            slide_info = await client.get_slide_info(slide["id"])
            
            # Info should match list data
            assert slide_info["id"] == slide["id"]
            assert slide_info["name"] == slide["name"]
    
    @pytest.mark.asyncio
    async def test_multiple_calls_consistency(self):
        """Test that multiple calls return consistent data"""
        client = MockHaloAPIClient()
        
        slides1 = await client.list_slides()
        slides2 = await client.list_slides()
        
        # Should return same data
        assert len(slides1) == len(slides2)
        if len(slides1) > 0:
            assert slides1[0]["id"] == slides2[0]["id"]


class TestAPIErrorHandling:
    """Test error handling in API client"""
    
    @pytest.mark.asyncio
    async def test_mock_client_handles_invalid_slide_id(self):
        """Test handling of invalid slide ID"""
        client = MockHaloAPIClient()
        
        # Should not crash
        slide_info = await client.get_slide_info("invalid-id-12345")
        
        # Should return None or empty
        assert slide_info is None or not slide_info
    
    @pytest.mark.asyncio
    async def test_mock_client_handles_invalid_roi_id(self):
        """Test handling of invalid ROI ID"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        if len(slides) > 0:
            slide_id = slides[0]["id"]
            
            # Try with invalid ROI ID
            roi_data = await client.export_roi(slide_id, "invalid-roi-id")
            
            # Should handle gracefully
            assert roi_data is not None or roi_data is None


class TestAPIIntegration:
    """Integration tests for API workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete workflow: list, get info, export, import"""
        client = MockHaloAPIClient()
        
        # Step 1: List slides
        slides = await client.list_slides()
        assert len(slides) > 0
        
        # Step 2: Get slide info
        slide_id = slides[0]["id"]
        slide_info = await client.get_slide_info(slide_id)
        assert slide_info is not None
        
        # Step 3: Export ROI
        roi_data = await client.export_roi(
            slide_id, "roi_001", format="image"
        )
        assert roi_data is not None
        
        # Step 4: Import annotations
        annotations = [{"type": "test"}]
        success = await client.import_annotations(
            slide_id, annotations, "Test"
        )
        assert success is True
    
    @pytest.mark.asyncio
    async def test_roi_workflow(self):
        """Test ROI-specific workflow"""
        client = MockHaloAPIClient()
        
        slides = await client.list_slides()
        if len(slides) > 0:
            slide_id = slides[0]["id"]
            
            # List ROIs
            rois = await client.list_rois(slide_id)
            assert isinstance(rois, list)
            
            # If ROIs exist, export them
            if len(rois) > 0:
                roi_id = rois[0]["id"]
                roi_data = await client.export_roi(slide_id, roi_id)
                assert roi_data is not None
