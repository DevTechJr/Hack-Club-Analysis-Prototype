# Slingshot Recommender (Hack Club Prototype)
### Take Home Project Submission - Anirudh

A Machine Learning based program that recommends students for a Slingshot Fellowship based on scraped web results from Hack Club's [online scrapbook](https://scrapbook.hackclub.com/).

### Project Description

Using scraped web results which have been stored in a csv, this program uses a trained model to determine whether or not the analysed student is fit for a fellowship at Slingshot. 
## Installation

Clone this repository and view "Hack Club Analysis.pynb" in Juptyer Notebook, VS Code or any other pynb viewing tool.

```bash
git clone https://github.com/DevTechJr/Hack-Club-Analysis-Prototype.git
```

## Tools & Library Imports

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
import joblib
from sklearn import tree
```
### Tools & Tech Specs

- Python 3.9
- Pandas
- Joblib
- SkLearn (Sci-kit learn)
- Jupyter Notebook
- CSV File (Data Storage)

## Data Cleaning Procedures

```python
# Data Cleaning
hackClub_data = pd.read_csv('cleanedHackClubData.csv')
hackClub_data = hackClub_data.dropna()
df = hackClub_data[['Streak','GHFollowers','GHFollowing','GHContributions','Result']]
df = df.dropna()
df = df.reset_index()
hackClub_data = hackClub_data.reset_index()
df['Streak'] = df['Streak'].map(lambda x: x.rstrip('-day streak'))
df['GHContributions'] = df['GHContributions'].map(lambda x: x.rstrip('\n      contributions\n        in the last year'))
df = df.replace(',','', regex=True)

# Final Data Processing & Creation
inputD = df.drop(columns=['Result'])
outputD = df['Result']
```
## Creating Model Persistence

```python
# Model Persistence
# joblib.dump(model,'slingshot-recommender.joblib')
model = joblib.load("slingshot-recommender.joblib")
modelInput = inputD
```

## Inital Test/Train Data Split & Model Creation

```python
# Split data into test/train sets
inputD_train,inputD_test,outputD_train,outputD_test = train_test_split(inputD,outputD,test_size=0.2)

# # Creating Model
model = DecisionTreeClassifier()
model.fit(inputD_train,outputD_train)
```

## Creating Model Persistence

```python
# Model Persistence
# joblib.dump(model,'slingshot-recommender.joblib')
model = joblib.load("slingshot-recommender.joblib")
modelInput = inputD
```

## Result Cleaning & Output

```python
# Predictions + Merging Results To A Table
predictions = model.predict(modelInput)
modelInput['Results'] = predictions

#Result Cleaning 
finalResults = pd.merge(modelInput, hackClub_data, how="left", on="index")
finalResults = finalResults.loc[:,~finalResults.columns.duplicated()].copy()
finalResults = finalResults.drop(columns=['Streak_x','GHFollowers_x','GHFollowing_x','GHContributions_x'])

# Data Output
finalResults

```

## Thoughts About This Project
- Lack of enough data for better accuracy (Because I used an online web scraper and it's free plan, my retrieved data was limited to 50 results only.)

- Web Scraper Program (I will write out a Scrapy script to automate the web scraping procedure without any dependency on other 3rd party tools as well as limitations on how many items I can scrap for free. The scraped web results were retrieved from a pre-made web scraper tool on a free plan. This resulted in a small data set size which surely would have influenced the results of the model and the bias in it.
)

- Random Forest Model (For the actual engine, I will switch to using a Random Forest Model to analyse data and determine results because of it's stronger accuracy and decision making, compared to the standard decision tree.)


## License
[MIT](https://choosealicense.com/licenses/mit/)
