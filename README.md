# Bank Customer Churn Prediction App

This application uses machine learning to predict whether a bank customer is likely to leave the bank (churn) or not based on various customer attributes.

## Demo

🚀 **App is now ready for deployment!** 

You can access the live demo of this application [here](#) (Link will be updated after Streamlit Cloud deployment).

![App Screenshot](https://placeholder.com/your-screenshot-after-deployment)

## Features

- Predict customer churn probability based on multiple factors
- Interactive form for entering customer data
- Visual representation of prediction results
- Risk factor analysis for high-churn-risk customers

## Dataset

The model was trained on the "Bank Customer Churn Modelling" dataset, which includes the following features:

- Credit Score
- Geography (France, Spain, Germany)
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary
- Churn Status (Target Variable)

## Model

The prediction model is an Artificial Neural Network (ANN) built with TensorFlow/Keras with the following architecture:

- Input Layer: Customer features (scaled and encoded)
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation (probability of churn)

## Local Setup

To run this application locally (the model is already trained):

### Step 1: Clone/Navigate to the repository
```bash
cd d:\annclassification
```

### Step 2: Create a virtual environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# For Mac/Linux users:
# source venv/bin/activate
```

### Step 3: Install required packages
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit app
```bash
streamlit run streamlit_app.py
```

### Step 5: Access the app
The app will automatically open in your browser at `http://localhost:8501`

### To deactivate the virtual environment (when done):
```bash
deactivate
```

## Quick Start Commands (Copy & Paste)

```bash
cd d:\annclassification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Troubleshooting

### Dependency Issues
If you encounter version conflicts with the requirements.txt file, try these alternatives:

**Option 1: Install with flexible versions**
```bash
pip install streamlit tensorflow pandas numpy scikit-learn matplotlib seaborn plotly joblib
```

**Option 2: Update pip first, then install**
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Option 3: Install specific working versions**
```bash
pip install streamlit>=1.28.0 tensorflow>=2.19.0 pandas>=1.5.0 numpy>=1.24.0 scikit-learn>=1.3.0
```

### Common Issues
- **TensorFlow version conflicts**: The app works with TensorFlow 2.19.0+ 
- **Python version**: Make sure you're using Python 3.8 or higher
- **Virtual environment**: Always use a virtual environment to avoid conflicts

## Deploying to Streamlit Cloud

### Prerequisites
✅ App runs locally  
✅ All required files are present  
✅ Requirements.txt is deployment-ready  

### Deployment Steps

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Ready for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repository
   - Select:
     - Repository: `your-repo-name`
     - Branch: `main`
     - Main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Deployment will complete in 2-3 minutes**

### Required Files for Deployment ✅
- ✅ `streamlit_app.py` - Main application
- ✅ `requirements.txt` - Dependencies  
- ✅ `model.h5` - Trained model
- ✅ `scaler.pkl` - Feature scaler
- ✅ `label_encoder_gender.pkl` - Gender encoder
- ✅ `onehot_encoder_geo.pkl` - Geography encoder

## Files in the Repository

- `streamlit_app.py`: The main Streamlit application
- `model.h5`: Trained neural network model
- `scaler.pkl`: StandardScaler for feature normalization
- `label_encoder_gender.pkl`: LabelEncoder for gender feature
- `onehot_encoder_geo.pkl`: OneHotEncoder for geography feature
- `Churn_Modelling.csv`: Dataset used for training
- `experiments.ipynb`, `prediction.ipynb`, etc.: Jupyter notebooks for model development and testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
