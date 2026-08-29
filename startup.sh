#!/bin/bash
echo "Starting LingAI..."
echo "Python version: $(python --version)"
echo "Checking model file..."
ls -lh sld.keras sld.weights.h5 || echo "Model files not found!"
echo "Launching Streamlit..."
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
