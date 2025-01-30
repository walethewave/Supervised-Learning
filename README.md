# Supervised Learning on House Rent Prediction

## Project Overview
This project utilizes supervised learning techniques to predict house rent prices based on various features such as the number of bedrooms, bathrooms, living area size, and location attributes. The dataset contains comprehensive details about properties, including structural features, geographic location, and sales data.

## Dataset Description
The dataset used for this project contains 4,600 records and 18 columns, including:

- **date:** Date of the property listing
- **price:** House price (target variable)
- **bedrooms:** Number of bedrooms
- **bathrooms:** Number of bathrooms
- **sqft_living:** Living area size in square feet
- **sqft_lot:** Lot size in square feet
- **floors:** Number of floors
- **waterfront:** Whether the property has a waterfront view (binary)
- **view:** Quality of the view from the property
- **condition:** Overall condition of the property
- **sqft_above:** Square footage above ground
- **sqft_basement:** Square footage of the basement
- **yr_built:** Year the property was built
- **yr_renovated:** Year the property was renovated
- **city:** City where the property is located
- **statezip:** State and ZIP code of the property
- **country:** Country of the property

## Key Data Preprocessing Steps

1. **Data Cleaning:**
    - Verified no missing values.
    - Converted the `date` column to datetime format and extracted relevant features such as `month` and `year`.

2. **Feature Engineering:**
    - Created a new feature `Age_of_the_House` as the difference between the current year and the year the house was built.
    - Dropped irrelevant columns (`country`, `street`, `date`, `year`).

3. **Encoding Categorical Data:**
    - Used `LabelEncoder` to transform categorical features such as `city` and `statezip`.

4. **Scaling:**
    - Applied `MinMaxScaler` to normalize numerical features.

## Model Architecture
A neural network model was built using TensorFlow/Keras:

- **Input Layer:** Dense layer with appropriate input shape.
- **Hidden Layers:** Multiple dense layers using ReLU activation.
- **Output Layer:** Dense layer with a single node for regression output.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

## Model Evaluation
The model was evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics:

- **MSE:** Indicates the average squared difference between predicted and actual prices.
- **MAE:** Provides the average magnitude of prediction errors.

## Results
The model demonstrated strong predictive capabilities, accurately capturing trends in house rent prices based on the given features.

## Visualizations
Key insights from the data were visualized using:
- Histograms and scatter plots for feature distributions.
- Correlation heatmaps to identify relationships between features.

```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```

## Conclusion
This project successfully demonstrates the application of supervised learning techniques for house rent prediction. With additional data and hyperparameter tuning, the model's performance can be further improved.

## Requirements
- Python 3.x
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## How to Run
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook or Python script.

## Author
Afolabi Olawale 
