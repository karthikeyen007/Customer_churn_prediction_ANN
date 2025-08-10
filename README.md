# Bank Customer Churn Prediction App

This application uses machine learning to predict whether a bank customer is likely to leave the bank (churn) or not based on various customer attributes.

## Demo

You can access the live demo of this application [here](#) (Link will be available after deployment).

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

To run this application locally:

1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/bank-customer-churn-prediction.git
   cd bank-customer-churn-prediction
   ```

2. Install the required packages
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run streamlit_app.py
   ```

## Deploying to Streamlit Cloud

1. Push your code to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app by connecting to your GitHub repository
4. Select the repository, branch, and the file `streamlit_app.py`
5. Click "Deploy"

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
