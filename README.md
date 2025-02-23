# Introduction to Machine Learning Summative

## Dataset

The data set is **sythesized** from the following sources:
- [National Bureau of Statistics (NBS)](https://www.nigerianstat.gov.ng/resource/SELECTED%20FOOD%20(JAN_2017%20-%20March%202022).xlsx)

The features seen in the dataset are engineered to be able to help SMEs in the food industry to predict the price of food items in the market.
It helps them to avoid wastage by **predicting if a Food item will sell out in a certain market session or not**. It is unique to Nigerian/African markets where "Night Market" (The cheapest market session) is a thing.

More information can be found in the [Notebook](./notebook.ipynb).

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
| Training Instance | Optimizer used(Adam, RMSPoP) | Regularizer Used(L1 and L2) | Epochs | Early Stopping(Yes or No) | Number of Layers | Learning Rate | Accuracy | F1 Score | Precision | Recall | Loss | Total +ve Predictions |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| Random Forest Classifier | N/A | N/A | N/A | N/A | N/A | N/A | 0.62 | 0.67 | 0.66 | 0.67 | N/A | 191 |
| Model 1 (Vanilla) | RMSprop | None | 45 | No | 4 | 0.001 | 0.65 | 0.69 | 0.70 | 0.69 | 0.64 | 202 |
| Model 2 (L2) | RMSprop | L2 | 500 | Yes | 4 | 0.00068 | 0.66 | 0.68 | 0.74 | 0.62 |0.65 | 205 |
| Model 3 (L1) | Adam | L1 | 500 | Yes | 4 | 0.0009 | 0.66 | 0.69 | 0.73 | 0.66 | 0.65 | 205 |
| Model 4 (Optimal) | Adam | L2 & L1 | 1000 | Yes | 4 | 0.00045 | 0.66 | 0.69 | 0.73 | 0.66 | 0.65 | 206 |

## Summary

### ML models vs Neural Network
---

#### Traditional ML models
Random Forest Classifiers, a popular choice among traditional ML models, are known for their robustness and ability to handle noisy data effectively. In our experiments, the Random Forest Classifier achieved an accuracy of 0.62, an F1 score of 0.67, and made 191 total positive predictions. Compared to the best performing neural network model (Model 4), the Random Forest Classifier had lower accuracy, F1 score, and total positive predictions, indicating that the neural network was able to capture more complex patterns and relationships in the data.

#### Neural Network
Neural networks, when architected and tuned appropriately, have the potential to outperform traditional ML models, even on smaller datasets. In our experiments, the best performing neural network model, Model 4 (Optimal), achieved an accuracy of 0.66, an F1 score of 0.69, and made 206 total positive predictions, surpassing the Random Forest Classifier. This demonstrates the superiority of the neural network in terms of accuracy, F1 score, and its ability to make more positive predictions compared to the traditional ML model.

### Neural Network comparison 
---

#### Model 1 (Vanilla)
Model 1, the vanilla neural network without any regularization, achieved an accuracy of 0.65, an F1 score of 0.69, and made 202 total positive predictions. The lack of regularization likely caused overfitting, where the model learned noise and specific patterns from the training data that did not generalize well to unseen data. This led to lower accuracy and fewer positive predictions compared to the regularized models.

#### Model 2 (L2)
Model 2 incorporated L2 regularization, which adds the squared magnitude of the weights to the loss function. L2 regularization encourages small but non-zero weights, effectively shrinking the weights of less important features. This helps prevent overfitting by controlling the model's complexity. In our experiments, Model 2 achieved an accuracy of 0.66 and made 205 total positive predictions. The increased precision (0.74) indicates that L2 regularization helped the model make more accurate positive predictions. However, the decreased recall (0.62) shows reduces effectiveness, missing some positive instances.

#### Model 3 (L1)
Model 3 employed L1 regularization, which adds the absolute values of the weights to the loss function. L1 regularization promotes sparsity by driving the weights of less important features to exactly zero. This feature selection property helps identify the most relevant features for the prediction task. Model 3 achieved an accuracy of 0.66, an F1 score of 0.69, and made 205 total positive predictions. The higher recall compared to Model 2 indicates that L1 regularization helped capture more positive instances. However, it slightly underperformed compared to Model 4, suggesting that combining L1 and L2 regularization could yield better results.

#### Model 4 (Optimal)
Model 4 utilized a combination of L1 and L2 regularization techniques, leveraging the strengths of both methods. L1 regularization helped identify the most informative features, while L2 regularization controlled the model's complexity and prevented overfitting. The combined effect allowed Model 4 to strike a balance between feature selection and weight regularization. As a result, Model 4 achieved the highest accuracy (0.66), F1 score (0.69), and total positive predictions (206) among the neural network models. The improved performance highlights the effectiveness of carefully combining regularization techniques to optimize model performance.

In conclusion, the choice of regularization technique significantly impacted the performance of the neural network models. L1 regularization helped with feature selection, L2 regularization controlled model complexity, and the combination of L1 and L2 in Model 4 provided the best balance, leading to superior performance. These results demonstrate the importance of selecting appropriate regularization methods and tuning hyperparameters to achieve optimal model performance.

## Video

[Watch the video explanation](https://drive.google.com/file/d/1S6Xd3CwgVb1CUTB0sddz8SHY168spUnD/view?usp=sharing)