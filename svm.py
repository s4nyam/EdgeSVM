import numpy as np
from sklearn.svm import SVR
from sklearn.externals import joblib

# Define the time series data
X = np.random.rand(100, 1)
y = np.roll(X, shift=-1)

# Define and train the SVM model
model = SVR(kernel='rbf', C=1e3, gamma=0.1)
model.fit(X, y)

# Make predictions using the trained model
X_test = np.random.rand(1, 1)
y_pred = model.predict(X_test)

# Save the trained model
joblib.dump(model, 'svm_model.pkl')

# Print the predicted value
print('Predicted value:', y_pred[0])
