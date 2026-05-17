# Health Insurance Premium Prediction

I built this project to explore how personal health and lifestyle factors — like age, BMI, and smoking habits — affect insurance costs. The goal was simple: can a machine learning model learn these patterns and make reasonable predictions?

**[Try the live app here](https://insurance-cost-prediction-cvyca9jnyndqwaqwzz4zxw.streamlit.app/)**

## What I did

I took a dataset of ~1,300 insurance records and built a Linear Regression pipeline from scratch. Instead of throwing everything into one script, I split the code into separate modules — data loading, preprocessing, visualization, and model training. Keeping things modular made it easier to debug and actually understand what each part was doing.

I also built a small web app with Streamlit so anyone can enter their details and get a cost estimate instantly — no code required.

## Results

The model ended up with an **R² score of 0.78**, meaning it explains about 78% of the variation in insurance costs. Not perfect, but honestly pretty solid for a linear model on this kind of data. The Mean Squared Error came in at **4181.19**.

One interesting thing I noticed: smoking status turned out to be by far the biggest factor in predicting costs — way more than age or BMI. That was a bit surprising at first, but makes sense when you think about it.

## Tech Stack

- Python
- Pandas, Scikit-Learn
- Matplotlib, Seaborn
- Joblib (for saving the model)
- Streamlit (for the web app)

## How to run it

pip install -r requirements.txt
python main.py
streamlit run app.py

## What I'd do differently next time

Linear Regression has its limits. I'd like to try Ridge Regression or even a Random Forest to see if the score improves — especially since the relationship between smoking and cost seems non-linear.

---

*Dataset from Kaggle — Medical Cost Personal Dataset*

---
**Developed by:** Maryam Lraimian