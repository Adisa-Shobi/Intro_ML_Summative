# Introduction to Machine Learning Summative

## Dataset

The data set is **sythesized** from the following sources:
- [National Bureau of Statistics (NBS)](https://www.nigerianstat.gov.ng/resource/SELECTED%20FOOD%20(JAN_2017%20-%20March%202022).xlsx)

The features seen in the dataset are engineered to be able to help SMEs in the food industry to predict the price of food items in the market.
It helps them to avoid wastage by **predicting if a Food item will sell out in a certain market session or not**. It is unique to Nigerian/African markets where "Night Market" (The cheapest market session) is a thing.

## How To Use
### Load the saved model
1. Load the model using the `joblib` library for traditional models or `tf.keras.models.load_model` for Keras models.
    ```python
    import joblib

    def make_predictions(model_path, X):

        # Check if the model is a Keras model or a traditional model
        if model_path.endswith('.keras'):
            # Load the Keras model
            model = tf.keras.models.load_model(model_path)
            # Make predictions
            predictions = model.predict(X)
        else:
            # Load the traditional model using joblib
            model = joblib.load(model_path)
            # Make predictions
            predictions = model.predict_proba(X)[:, 1]  # Assuming binary classification

        return predictions
    ```
2. Replace the model_path var with the name of the model you want to use. Make predictions using the `make_predictions` function.
    ```python
    # Load the model
    model_path = 'path/to/model.pkl'
    predictions = make_predictions(model_path, X)
    ```

## Model Stats
| Training Instance | Optimizer used(Adam, RMSPoP) | Regularizer Used(L1 and L2) | Epochs | Early Stopping(Yes or No) | Number of Layers | Learning Rate | Accuracy | F1 Score | Precision | Recall | Loss |
| - | - | - | - | - | - | - | - | - | - | - | - |
| Random Forest Classifier | N/A | N/A | N/A | N/A | N/A | N/A | 0.63 | 0.67 | 0.65 | 0.68 | N/A |
| Model 1 (Vanilla) | RMSprop | None | 300 | No | 4 | 0.001 | 0.59 | 0.54 | 0.69 | 0.44 | 0.67 |
| Model 2 (L2) | RMSprop | L2 | 500 | Yes | 4 | 0.000059 | 0.58 | 0.60 | 0.63 | 0.57 |11.07 |
| Model 3 (L1) | Adam | L1 | 500 | Yes | 4 | 0.000018 | 0.59 | 0.67 | 0.60 | 0.75 | 13.17 |
| Model 4 (Optimal) | Adam | L2 | 1000 | Yes | 4 | 0.0001 | 0.62 | 0.65 | 0.65 | 0.64 | 1.08 |
## Summary

### ML models vs Neural Network

#### Traditional ML model
The ML models overall had a better performance than the Neural Network models. The Random Forest Classifier had the best performance with an accuracy of 0.63 and an F1 score of 0.67. The Neural Network models had an accuracy of 0.59, 0.58, 0.59, and 0.62 for Model 1, Model 2, Model 3, and Model 4 respectively. The F1 score for the Neural Network models were 0.54, 0.60, 0.67, and 0.65 for Model 1, Model 2, Model 3, and Model 4 respectively. 








#### Neural Network

The numbers between the last neural network instance and the Random Forest Classifier are very close. Neural networks are known to be good when dealing with large datasets 

- The Random Forest Classifier has a most of its optimizations done under the hood once the hyper parameters are tuned. Beyond this there is not much tweak whereas Nueral network are highly configurable and have a non exhaustive list of hyper parameters to tune. This is as good as it gets for the Random Forest Classifier.

### Neural Network comparison

#### Model 1 (Vanilla)

#### Model 2 (L2)

#### Model 3 (L1)

#### Model 4 (Optimal)