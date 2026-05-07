import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

class InsuranceModel:
     def __init__(self):
            self.model = LinearRegression()

     def train(self,X_train,y_train):
           self.model.fit(X_train,y_train)
           print('we do that successefully')

     def evaluate(self,X_test,y_test):
           predictions = self.model.predict(X_test) 
           mse = mean_absolute_error(y_test,predictions)
           r2 = r2_score(y_test,predictions)
           return mse,r2

     def save_model(self,file_path):
           """Saves the trained model to a file."""
           joblib.dump(self.model,file_path)
           print(f'Model saved to {file_path}')

     def load_model(self,file_path):
          """Loads a model from a file."""
          self.model = joblib.load(file_path)
          print(f"Model Load From:{file_path}")

           
            
                 