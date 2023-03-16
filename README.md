# Generic-Naive-Bayes

![](https://miro.medium.com/max/702/0*3_J7YH5beFVmpxBg.png)

## Description
This is an implementation of a Naive Bayes Classifier algorithm that predicts the class label of an input sequence based on its features. The class is initialized with training data, labels, and a smoothing parameter, and it uses Bayes' theorem to calculate the conditional probability of a class given a feature.


## How It Works

## Dependencies
This package uses the following libraries.
* Python 3.8
* numpy

## Installing and Executing program

1. Clone repository
    ```
    git clone git@github.com:logic-ot/Generic-Naive-Bayes.git
    ```
2. Installing Depedencies
    ```
    virtualenv myenv
    myenv\scripts\activate    
    pip install -r requirements.txt
    ```
2. Import the model
   ```
   from generic_bayes import bayesian_classifier
   ```
3. Instantiate the model. This automatcailly fits the training data to the model
    ```
    model = bayesian_classifier(X_train,Y_train)
    ```
5. Use the fitted model to predict a class using a sequence of features 
    ```
    model.predict(X_target.iloc[10])
    ```
### Example code
    from generic_bayes import bayesian_classifier
    import numpy as np

    model = bayesian_classifier(np.array([[1,1],[2,1],[3,2]]), np.array([1,2,2]))

    print(model.predict(np.array([1,1,1,1,1,3,1])))

    OUTPUT: [1]

    
  ## Limitations


