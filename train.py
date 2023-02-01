import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
###Dataset 
data = pd.read_json("/content/drive/MyDrive/Sarcasm.json", lines=True)
print(data.head())
#### Information abt dataset
data.info()
#ploting the label column
data['is_sarcastic'].value_counts()

import seaborn as sns
sns.set(rc={'figure.figsize':(8,8)})
sns.countplot(data['is_sarcastic'])
#word visualization for sarcastic class
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def make_wordcloud(words,title):
    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white").generate(words)
    plt.figure(figsize=(20,20))
    plt.imshow(cloud, interpolation="gaussian")
    plt.axis("off") 
    plt.title(title, fontsize=60)
    plt.show()
    
all_text = " ".join(data[data.is_sarcastic == 1].headline) 
make_wordcloud(all_text, "Sarcasm")
#word visualization for non sarcastic class
all_text = " ".join(data[data.is_sarcastic == 0].headline) 
make_wordcloud(all_text, "Non_Sarcasm")
#install the lazy text predict library
pip install lazy-text-predict

X = data.headline
Y = data.is_sarcastic
# perform train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
#initialize the model and train
from lazytextpredict import basic_classification
trial=basic_classification.LTP(Xdata=X,Ydata=Y, models='all')
trial.run(training_epochs=5)
#print the metrics
trial.print_metrics_table()

