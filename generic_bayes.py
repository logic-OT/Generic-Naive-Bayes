import numpy as np 

class bayesian_classifier:
    def __init__(self,x: np.ndarray, y:np.ndarray, alpha = 1):
        '''
        Notes
        ----------
        This module is a simple implemetation of Multi-Nomial Naive Bayes. You can easily inset any discrete
        data along with target values and quickly make predictions

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            where n_samples is the number of samples and n_features is the number of features.

        y : ndarray of shape (n_samples,)
            Target values.

        '''
        self.alpha = alpha
        unique_classes = np.unique(y)
        priors = {}
        for i in unique_classes:
            priors[i] = len(y[y == i])/len(y)
        #check whether x is discrete
        self.__features = x
        self.__classes = y
        self.__priors = priors
        self.__unique_classes = unique_classes
    
    def __bayesian_prob(self,label,feature):
            #bayes' theorem is given by P(C|F) = (P(F|C)*P(C))/P(F)
            #where C = Classes/labels and F = features

            #P(F|C)*P(C)
            stripped = self.__features[self.__classes==label]
            instances = len(stripped[stripped == feature])+self.alpha
            likelihood = instances/len(stripped)
                
            return (likelihood*self.__priors[label])


    def predict(self, Input ):
        '''
        Parameters
        ----------
        Input: 1D ndarray (of shape (n_samples,))
            Sequence to predict (list of features)

        Returns
        -------
        np.ndarray
            The class the input sequence belongs to
        '''
        # for each feature in the Input parameter, we calculate prob of getting the various classes given the feature.
        # example: if one feature(F) is in the input, and there are 2 classes C1 and C2, we calculate, 
        # P(C1|A) and P(C2|A) and select the highest
        
        classes_given_features = []
        for label in self.__unique_classes:
            features_given_class = list(self.__bayesian_prob(label,value) for value in Input)
            features_given_class = np.prod(features_given_class)
            classes_given_features.append(features_given_class)

        return([self.__unique_classes[np.argmax(classes_given_features)]])
        

        

