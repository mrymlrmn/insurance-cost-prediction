import os
from src.preprocessing import InsurancePreprocessing
from src.model_train import InsuranceModel
from src.data_loader import load_insurance_data
from src.visualizer import InsuranceVisualizer

def main():

    # 1. Load
    raw_data = load_insurance_data('data/insurance.csv')

    # 2. Visualizing Data
    visualizer = InsuranceVisualizer(raw_data)
    visualizer.full_report()
    # 3. Initialize Preprocessor with the data
    preprocessor = InsurancePreprocessing(raw_data)

    #4.Process
    preprocessor.clean_data()
    preprocessor.encode_data()

    # 5.Split Data into Train and Test sets 
    # This creates X_train, X_test, y_train, y_test
    X_train,X_test,y_train,y_test = preprocessor.split_data()
    print(f"Training Sample:{len(X_train)}")
    print(f"Testing Sample:{len(X_test)}")

    trainer = InsuranceModel()
    trainer.train(X_train,y_train)
    mse , r2 = trainer.evaluate(X_test,y_test)
    print("-"*30)
    print(f"Model Performane Metrics:")
    print(f"Mean Squared Error:{mse:.2f}")
    print(f"R2 Score(Acuracy):{r2:0.4f}")

    # 6. Save the Brain (Model)
    if not os.path.exists('models'):
        os.makedirs('models')
    trainer.save_model('models/insurance_model.pkl')


if __name__ == "__main__":
    main()