#!/bin/bash
# Demo script to launch the Halo AI Workflow web application

echo "=================================================="
echo "   Halo AI Workflow - Digital Pathology Analysis"
echo "=================================================="
echo ""
echo "Starting web application..."
echo ""
echo "Once launched, open your browser to: http://localhost:8501"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

# Launch Streamlit app
streamlit run app.py --server.headless=true --server.port=8501 --server.address=localhost
