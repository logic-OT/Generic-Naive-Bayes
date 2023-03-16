# Generic-Naive-Bayes

![](https://miro.medium.com/max/702/0*3_J7YH5beFVmpxBg.png)

## Description
This is an implementation of a <b>Multinomial Naive Bayes Classifier algorithm</b> that predicts the class label of an input sequence based on its features. The code initializes training data, labels, and a smoothing parameter, and it uses <b>Bayes' theorem</b> to calculate the conditional probability of a class given a feature. Since its multinomial, the data used should be discrete in nature


## How It Works
The code defines a class called bayesian_classifier which implements a simple Naive Bayesian Classifier algorithm. The class is initialized with training data, labels, and a smoothing parameter, and it has a predict method that takes an input sequence and returns the predicted class label based on the features of the sequence.

The <b>__init__ method</b> initializes the object with the input training data and labels, and it calculates the prior probabilities of each class from the input training data. 
The <b>__bayesian_prob method</b> calculates the conditional probability of a class given a feature using Bayes' theorem.

The <b>predict method</b> calculates the likelihood of the input sequence belonging to each class by multiplying the conditional probabilities of each feature given the class, and it returns the class with the highest likelihood as the predicted label.

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
3. Import the model
   ```
   from generic_bayes import bayesian_classifier
   ```
4. Instantiate the model and pass in your data. This automatcailly fits the training data to the model
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

- Doesn't have a system to detect continuous data and throw an error
