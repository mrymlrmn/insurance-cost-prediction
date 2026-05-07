import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceVisualizer:
    
    def __init__(self,df):
        self.df = df
        sns.set_theme(style='whitegrid')

    def full_report(self):
        fig,axes =plt.subplots(1,3,figsize=(20,6))
        fig.suptitle('Exploratory Data Analysis (EDA)', fontsize = 16)
    
        sns.histplot(self.df['age'],kde=True,ax=axes[0], color='salmon')
        axes[0].set_title('Age Distribution')
          
        sns.boxplot(x='smoker',y='charges',data=self.df,ax=axes[1])
        axes[1].set_title('Impact of Smoking on Charges')
        
        numeric_df = self.df.select_dtypes(include=['number'])
        sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm',ax=axes[2])
        axes[2].set_title('Correlation Matrix')

        plt.tight_layout()
        plt.show()


    
