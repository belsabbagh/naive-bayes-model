import pandas as pd
from src.nb import NaiveBayes


if __name__ == '__main__':
    nb = NaiveBayes()
    df = pd.read_csv('data/tennis.csv')
    nb.fit(df[['Outlook','Temperature','Humidity','Windy']], df['Play'])
    res = [nb.predict(row) for _, row in df[['Outlook','Temperature','Humidity','Windy']].iterrows()]
    print(res)
    