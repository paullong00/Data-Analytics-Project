# -*- coding: utf-8 -*-
"""
Name: Paul Long

"""

import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer


def Task1():
    #20 Oldest Athletes by Country
    try:
        data = pd.read_csv('athlete_events.csv')

        top_20_oldest = data.sort_values(by='Age', ascending=False).head(20)
        
        country_counts = top_20_oldest['Team'].value_counts()

        plt.figure(figsize=(8, 8))
        plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Percentage of Athletes from Each Country (Top 20 Oldest)')
        plt.axis('equal')

        plt.show()
    
    except FileNotFoundError:
        print("Error: The 'athlete_events.csv' file was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The 'athlete_events.csv' file is empty.")
    except KeyError as e:
        print(f"Error: The column '{e.args[0]}'")
    except ValueError as e:
        print(f"Error: There is a problem with the cell format. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    


def Task2():
    #Height Distribution of Athletes
    
    data = pd.read_csv('athlete_events.csv')

    plt.figure(figsize=(8, 6))
    plt.boxplot(data['Height'], vert=False)
    plt.title('Height Distribution Amoung Athletes')
    plt.xlabel('Height (cm)')
    
    median_height = data['Height'].median()
    max_height = data['Height'].max()
    min_height = data['Height'].min()

    q1 = data['Height'].quantile(0.25)
    q3 = data['Height'].quantile(0.75)
    iqr = q3 - q1
    upper_whisker = q3 + 1.5 * iqr
    lower_whisker = q1 - 1.5 * iqr
    
    outliers = data[(data['Height'] < lower_whisker) | (data['Height'] > upper_whisker)]
    plt.scatter(outliers['Height'], [1]*len(outliers), color='orange', marker='x', label='Outliers')

    plt.scatter(median_height, 1, color='red', marker='o', label='Median')
    plt.scatter(max_height, 1, color='green', marker='^', label='Max')
    plt.scatter(min_height, 1, color='blue', marker='v', label='Min')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    


def Task3():
    #Top 10 Countries by Gold Medals Won
    
    data = pd.read_csv('athlete_events.csv')

    gold_data = data[data['Medal'] == 'Gold']

    gold_medals_by_country = gold_data['Team'].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    plt.bar(gold_medals_by_country.index, gold_medals_by_country.values)
    plt.title('Top 10 Countries with the Maximum Number of Gold Medals')
    plt.xlabel('Country (Team)')
    plt.ylabel('Number of Gold Medals')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    


def Task4():
    #Predict Medal By Groupings of 4
    
    data = pd.read_csv('athlete_events.csv')
    
    data = data.dropna(subset=['Medal'])
    encoder = LabelEncoder()
    data['Sex'] = encoder.fit_transform(data['Sex'])
    data['Team'] = encoder.fit_transform(data['Team'])
    data['Sport'] = encoder.fit_transform(data['Sport'])

    data['Year'] = data['Games'].apply(lambda x: int(x[:4]))

    imputer = SimpleImputer(strategy='mean')
    data[['Height', 'Weight']] = imputer.fit_transform(data[['Height', 'Weight']])

    all_features = ['Sex', 'Height', 'Team', 'Weight', 'Sport', 'Year']
    results = []
    feature_combinations = list(combinations(all_features, 4))
    X = data[list(all_features)]
    y = data['Medal']
    
    for features in feature_combinations:
        X_subset = X[list(features)]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=1)
        model = LogisticRegression(max_iter=271117, random_state=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((features, accuracy))
    
    results.sort(key=lambda x: x[1], reverse=True)

    feature_names = [', '.join(features) for features, _ in results]
    accuracies = [accuracy for _, accuracy in results]
    
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, accuracies)
    plt.xlabel('Feature Group')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.title('Accuracy of Feature Groups in Predicting Medal')
    plt.tight_layout()
    plt.show()


    

def Task5():
    #Sports in which Women got more Gold Medals than Men

    data = pd.read_csv('athlete_events.csv')

    gold_data = data[data['Medal'] == 'Gold']

    gold_medals_by_sport_sex = gold_data.groupby(['Sport', 'Sex']).size().unstack(fill_value=0)

    sports_women_outperformed_men = gold_medals_by_sport_sex[gold_medals_by_sport_sex['F'] > gold_medals_by_sport_sex['M']].index

    print("Sports where women received more Gold medals than men:")
    for sport in sports_women_outperformed_men:
        print(sport)
    
    


def Task6():
    #Years with Maximum Number of each medal
    
    data = pd.read_csv('athlete_events.csv')

    gold_medals_by_year = data[data['Medal'] == 'Gold'].groupby('Year').size()
    silver_medals_by_year = data[data['Medal'] == 'Silver'].groupby('Year').size()
    bronze_medals_by_year = data[data['Medal'] == 'Bronze'].groupby('Year').size()

    year_with_max_gold = gold_medals_by_year.idxmax()

    year_with_max_silver = silver_medals_by_year.idxmax()

    year_with_max_bronze = bronze_medals_by_year.idxmax()

    print("Year with the maximum number of Gold medals:", year_with_max_gold)
    print("Year with the maximum number of Silver medals:", year_with_max_silver)
    print("Year with the maximum number of Bronze medals:", year_with_max_bronze)
    
Task1()
Task2()
Task3()
Task4()
Task5()
Task6()
