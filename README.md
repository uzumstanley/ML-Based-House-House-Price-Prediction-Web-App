# ML-Based House Price Prediction Web App

**1. Introduction**

Data scientists can leverage web frameworks like Flask to deploy machine learning models, allowing non-technical users to interact with the models through an intuitive web interface. This report demonstrates the use of Flask to serve a house price prediction model, trained on real estate data. The primary objective is to showcase how models can be integrated into web applications to provide real-time predictions and enhance business value.

**1.1. Problem Statement**

Accurately predicting house prices based on features such as square footage, the number of bedrooms, the year the house was built, and other relevant characteristics is critical for homeowners, real estate agents, and potential buyers. This project involves building a regression model to predict house prices and deploying it as a web-based tool. 

**2. Dataset Overview**

The dataset consists of 21,613 records with 19 features, which include various characteristics of properties in a region. The target variable is the house price. The key features used in the model include:

•	bedrooms: Number of bedrooms.

•	sqft_living: Square footage of living space.

•	condition: The overall condition of the house.

•	yr_built: Year the house was built.

•	yr_renovated: Year of renovation.


Data link here – [houseprice_data.csv](https://github.com/UdonnaM/ML-Based-House-Price-Prediction-Web-App/blob/main/houseprice_data.csv)


**3. Model Building and Evaluation**

After loading the dataset, we initially had 19 columns, representing various features related to house characteristics, neighborhood averages, and property sizes. These include features such as the number of bedrooms, bathrooms, square footage of living space (sqft_living), and lot size (sqft_lot), among others.

**3.2 Feature Selection and Column Reduction**

During the preprocessing phase, we noticed a reduction in the number of columns from 19 to 17. Upon investigation, we determined that the following columns were excluded:

i.	sqft_above: Represents the square footage of the house excluding the basement.

ii. sqft_living15: Represents the average square footage of living space in the neighborhood (within a 15-house radius).
   
**Reason for Removal:**

The exclusion was due to multicollinearity—where certain features are highly correlated with each other. For instance:

•	sqft_living and sqft_living15: Both describe similar characteristics of house size, with one referring to the specific house and the other to the neighborhood average.

•	sqft_above: Highly correlated with sqft_living, which already encapsulates total living space.

By removing these correlated features, the model can improve its generalization ability, avoiding overfitting to redundant information.

**3.3 Model Training**

We used a Multiple Linear Regression model to predict house prices based on the remaining 17 features. The model was trained on the processed dataset, which had 20,322 rows and 17 columns after removing missing or irrelevant data points.

Model Parameters:

The main features used in the model are:

•	Number of bedrooms (bedrooms)

•	Square footage of living space (sqft_living)

•	Condition of the house (condition)

•	Year built (yr_built)

•	Year renovated (yr_renovated)

**3.4 Model Results**

After training the model, the following results were obtained:

•	Coefficients: The multiple linear regression model produced the following coefficients for the selected features: -54727.93 (for bedrooms), 326.47 (for square footage of living space), 21645.77 (for condition), -1997.91 (for year built), and 33.93 (for year renovated). These values represent the weight or impact each feature has on the predicted house price. For example, for every additional square foot of living space, the price is expected to increase by approximately $326.47, holding other factors constant.

•	Intercept: The intercept is 3,906,669.75. This represents the base price when all features are zero, although in practice, it serves as an adjustment factor in the model’s prediction equation.

•	Mean Squared Error (MSE): The model produced a mean squared error of 64,139,982,898.92. This metric reflects the average squared difference between the actual and predicted house prices, with lower values indicating better model performance.

•	Coefficient of Determination (R²): The R² score is 0.56. This value indicates that the model explains 56% of the variance in house prices. While this suggests moderate accuracy, there is room for improvement, and including other features or exploring more complex models may lead to higher predictive power.

•	Model Persistence: The trained model was saved as 'Multiplelinear_regression_model.pkl' for future reuse.

Model training code link: [MultipleLinearRegression3dcode-1 - app.py](MultipleLinearRegression3dcode-1%20-%20app.py)

**4. Model Deployment Using Flask**

**4.1 Flask Application Overview**

To demonstrate how a data scientist can leverage web frameworks for machine learning models, we used Flask to deploy the house price prediction model as a web application.

**4.2 Application Structure**

The Flask web application allows users to input features like the number of bedrooms, square footage, condition, year built, and year renovated to predict the house price. The steps involved:

i.	Model Loading: The saved model (Multiplelinear_regression_model.pkl) is loaded into the Flask app.

ii.	User Inputs: The web form captures user inputs for features required by the model.

iii.	Prediction: After receiving input, the app uses the model to generate a prediction.

iv.	Output: The predicted house price is displayed on the web page.

Application Files:

•	app.py: The main Flask application file. Code link here: [app.py](app.py)

•	HTML Templates: The web page templates for user interaction and result display. Code link here: [index.html](index.html)

**4.3 Deployment Process**

The Flask application was deployed locally for testing. The following steps were taken:

i.	Local Environment Setup: Flask and required dependencies were installed, and the app was run locally to ensure it worked as expected.

ii.	Testing the API: The web app was tested with various inputs, verifying that it correctly displayed the predicted house prices based on the model’s output.

**Screenshot of User Interface**

![House Price Prediction Screenshot](Screenshots/house%20price%20prediction.png)


**5. Conclusion**

In this report, we built a house price prediction model using Multiple Linear Regression. Despite using only a subset of the available features, the model achieved an R² score of 0.56. While this performance is moderate, it demonstrates the power of simple linear models in solving real-world regression tasks.

The Flask web application showed how easily data scientists can deploy machine learning models as interactive web services. Such applications allow non-technical users to leverage predictive models without needing direct access to the code or data.

Future Enhancements
To improve the model and application further:

i.	Explore Non-Linear Models: Implementing more advanced models, such as decision trees or gradient boosting, may improve accuracy.

ii.	Feature Engineering: Adding new features or transforming existing ones could enhance the model’s predictive power.

iii.	Deploying to the Cloud: The application can be deployed on platforms like Heroku or AWS, making it publicly accessible and scalable.

