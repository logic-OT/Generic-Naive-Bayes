from generic_bayes import bayesian_classifier
import numpy as np

model = bayesian_classifier(np.array([[1,1],[2,1],[3,2]]), np.array([1,2,2]))

print(model.predict(np.array([1,1,1,1,1,3,1])))
