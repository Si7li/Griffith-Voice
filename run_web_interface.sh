#!/bin/bash

# Real-time Voice Translation Web Interface Launcher
# This script activates the virtual environment and starts the Streamlit app

echo "🗣️ Starting Real-time Voice Translation Web Interface..."
echo "======================================================"

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv env"
    echo "   source env/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source env/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit streamlit-player
fi

# Set environment variables to reduce warnings
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore::UserWarning"
export TORCHAUDIO_BACKEND=soundfile

# Start the Streamlit app
echo "🚀 Starting web interface..."
echo "📱 Open your browser and go to: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Use the clean version with better warning handling
streamlit run streamlit_webui.py --server.port 8501 --server.address 0.0.0.0 --logger.level error
