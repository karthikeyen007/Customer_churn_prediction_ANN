# Create a Python virtual environment if it doesn't exist
if (-not (Test-Path -Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate the virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
Write-Host "Starting Streamlit app..."
streamlit run streamlit_app.py

# Note: To deactivate the virtual environment when done, run:
# deactivate
