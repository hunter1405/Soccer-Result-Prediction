# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 19:33:11 2022

@author: ACER
"""
# Import librarbies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#*************************************************************************#

#Insert path here
f1 = 'C:/Users/ACER/Downloads/data_csv/match.csv'
f2 = 'C:/Users/ACER/Downloads/data_csv/team.csv'
f3 = 'C:/Users/ACER/Downloads/data_csv/team_attributes.csv'
f4 = 'C:/Users/ACER/Downloads/data_csv/player_attributes.csv'
f5 = 'C:/Users/ACER/Downloads/data_csv/player.csv'
f6 = 'C:/Users/ACER/Downloads/data_csv/country.csv'
f7 = 'C:/Users/ACER/Downloads/data_csv/league.csv'

#*************************************************************************#

# Load required data
matches_df = pd.read_csv(f1,  header = 0, sep ='\t')
teams_df = pd.read_csv(f2,  header = 0, sep ='\t')
player_attributes_df = pd.read_csv(f4,  header = 0, sep ='\t')

sample = matches_df.head(100)
sample2 = teams_df.sample(100)
sample3 = player_attributes_df.sample(100)

#*************************************************************************#
# Data Preparation

#Convert Date to DateTime type to gain availability to dedicate a year of each date
matches_df['date'] = pd.to_datetime(matches_df['date'], format='%Y-%m-%d 00:00:00')

# Select columns which we need in analysis
home_players = ["home_player_" + str(x) for x in range(1, 12)]
away_players = ["away_player_" + str(x) for x in range(1, 12)]

matches_kept_columns = ["id","match_api_id", "date", "home_team_api_id", "away_team_api_id", 
                        "home_team_goal", "away_team_goal"]
matches_kept_columns = matches_kept_columns + home_players
matches_kept_columns = matches_kept_columns + away_players

matches_df = matches_df[matches_kept_columns]

# Check Null
matches_df.isnull().any().any(), matches_df.shape
how_many_null = matches_df.isnull().sum(axis=0)
matches_df = matches_df.dropna()
matches_df.isnull().any().any(), matches_df.shape

# Add label columns to match table
matches_df['goal_difference'] = matches_df['home_team_goal'] - matches_df['away_team_goal']
matches_df['home_status'] = 'D'
matches_df['home_status'] = np.where(matches_df['goal_difference'] > 0, 'HW', matches_df['home_status'])
matches_df['home_status'] = np.where(matches_df['goal_difference'] < 0, 'HL', matches_df['home_status'])

# Merge new features with the initial dataframe
for player in home_players:
    matches_df = pd.merge(matches_df, player_attributes_df[["id", "overall_rating"]], 
                          left_on=[player], right_on=["id"], suffixes=["", "_" + player])
for player in away_players:
    matches_df = pd.merge(matches_df, player_attributes_df[["id", "overall_rating"]], 
                          left_on=[player], right_on=["id"], suffixes=["", "_" + player])
    
matches_df = matches_df.rename(columns={"overall_rating": "overall_rating_home_player_1"})

matches_df = matches_df[ matches_df[['overall_rating_' + p for p in home_players]].isnull().sum(axis = 1) <= 0]
matches_df = matches_df[ matches_df[['overall_rating_' + p for p in away_players]].isnull().sum(axis = 1) <= 0]

matches_df['overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].sum(axis=1)
matches_df['overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].sum(axis=1)
matches_df['overall_rating_difference'] = matches_df['overall_rating_home'] - matches_df['overall_rating_away']

matches_df['min_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].min(axis=1)
matches_df['min_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].min(axis=1)

matches_df['max_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].max(axis=1)
matches_df['max_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].max(axis=1)

matches_df['mean_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].mean(axis=1)
matches_df['mean_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].mean(axis=1)

matches_df['std_overall_rating_home'] = matches_df[['overall_rating_' + p for p in home_players]].std(axis=1)
matches_df['std_overall_rating_away'] = matches_df[['overall_rating_' + p for p in away_players]].std(axis=1)

# Drop unnecessary columns
# Player's rating
for c in matches_df.columns:
    if '_player_' in c:
        matches_df = matches_df.drop(c, axis=1)

# Filter the latest dates of each unique matches
matches_df = matches_df.sort_values(["match_api_id","date"])
matches_df = matches_df.drop_duplicates("match_api_id", keep="last")

# Merge home and away team's names
team_home = teams_df.rename (columns={"team_api_id": "home_team_api_id", 
                                      'team_long_name': "home_team_name"})
matches_df = pd.merge(matches_df, team_home[["home_team_api_id","home_team_name"]],
                      how="left", on=["home_team_api_id"])

team_away = teams_df.rename(columns={"team_api_id": "away_team_api_id",
                                  "team_long_name": "away_team_name"})
matches_df = pd.merge(matches_df, team_away[["away_team_api_id", "away_team_name"]],
                how='left',on=["away_team_api_id"])     

#*************************************************************************#
# Exploratory Data Analysis
df = matches_df.copy()

# Overall_rating
df = matches_df.copy()
def overall_rating(df):
    df["compare_overall_rating"] = "home_is_lower"
    df.loc[df["overall_rating_home"] > df["overall_rating_away"],
           "compare_overall_rating"] = "home_is_higher"
    df.loc[df["overall_rating_home"] == df["overall_rating_away"],
           "compare_overall_rating"] = "home_is_equal"
    
    df["compare_overall_rating"].value_counts()
    check = df.groupby("compare_overall_rating")["home_status"].value_counts().unstack()
    check["win_rate"] = check["HW"] / (check["HW"] + check["HL"] + check["D"])
    return check

compare_overall_rating = overall_rating(df)
compare_overall_rating['win_rate'] = pd.Series(["{0:.2f}%".format(val * 100) for val in compare_overall_rating['win_rate']],
                                               index = compare_overall_rating.index)

# Overall_rating_difference
sns.set_theme(style="whitegrid")
sns.boxplot(x="home_status", y="overall_rating_difference", data=df, fliersize=0).set(
    xlabel='Match Result', 
    ylabel='Overall Rating Difference',
    ylim=(-100, 100))

# Min_overall_rating
def min_overall_rating(df):
    df["compare_min_overall_rating"] = "home_is_lower"
    df.loc[df["min_overall_rating_home"] > df["min_overall_rating_away"],
           "compare_min_overall_rating"] = "home_is_higher"
    df.loc[df["min_overall_rating_home"] == df["min_overall_rating_away"],
           "compare_min_overall_rating"] = "home_is_equal"

    df["compare_min_overall_rating"].value_counts()
    check = df.groupby("compare_min_overall_rating")["home_status"].value_counts().unstack()
    check["win_rate"] = check["HW"] / (check["HW"] + check["HL"] + check["D"])
    return check

compare_min_overall_rating = min_overall_rating(df)
compare_min_overall_rating['win_rate'] = pd.Series(["{0:.2f}%".format(val * 100) for val in compare_min_overall_rating['win_rate']],
                                               index = compare_min_overall_rating.index)

# Max_overall_rating
def max_overall_rating(df):
    df["compare_max_overall_rating"] = "home_is_lower"
    df.loc[df["max_overall_rating_home"] > df["max_overall_rating_away"],
           "compare_max_overall_rating"] = "home_is_higher"
    df.loc[df["max_overall_rating_home"] == df["max_overall_rating_away"],
           "compare_max_overall_rating"] = "home_is_equal"

    df["compare_max_overall_rating"].value_counts()
    check = df.groupby("compare_max_overall_rating")["home_status"].value_counts().unstack()
    check["win_rate"] = check["HW"] / (check["HW"] + check["HL"] + check["D"])
    return check

compare_max_overall_rating = max_overall_rating(df)
compare_max_overall_rating['win_rate'] = pd.Series(["{0:.2f}%".format(val * 100) for val in compare_max_overall_rating['win_rate']],
                                               index = compare_max_overall_rating.index)

# Mean_overall_rating
def mean_overall_rating(df):
    df["compare_mean_overall_rating"] = "home_is_lower"
    df.loc[df["mean_overall_rating_home"] > df["mean_overall_rating_away"],
           "compare_mean_overall_rating"] = "home_is_higher"
    df.loc[df["mean_overall_rating_home"] == df["mean_overall_rating_away"],
           "compare_mean_overall_rating"] = "home_is_equal"

    df["compare_mean_overall_rating"].value_counts()
    check = df.groupby("compare_mean_overall_rating")["home_status"].value_counts().unstack()
    check["win_rate"] = check["HW"] / (check["HW"] + check["HL"] + check["D"])
    return check

compare_mean_overall_rating = mean_overall_rating(df)
compare_mean_overall_rating['win_rate'] = pd.Series(["{0:.2f}%".format(val * 100) for val in compare_mean_overall_rating['win_rate']],
                                               index = compare_mean_overall_rating.index)
# Standardd_deviation_overall_rating
def std_overall_rating(df):
    df["compare_std_overall_rating"] = "home_is_lower"
    df.loc[df["std_overall_rating_home"] > df["std_overall_rating_away"],
           "compare_std_overall_rating"] = "home_is_higher"
    df.loc[df["std_overall_rating_home"] == df["std_overall_rating_away"],
           "compare_std_overall_rating"] = "home_is_equal"

    df["compare_std_overall_rating"].value_counts()
    check = df.groupby("compare_std_overall_rating")["home_status"].value_counts().unstack()
    check["win_rate"] = check["HW"] / (check["HW"] + check["HL"] + check["D"])
    return check

compare_std_overall_rating = std_overall_rating(df)
compare_std_overall_rating = compare_std_overall_rating.fillna(0)
compare_std_overall_rating['win_rate'] = pd.Series(["{0:.2f}%".format(val * 100) for val in compare_std_overall_rating['win_rate']],
                                               index = compare_std_overall_rating.index)

#*************************************************************************#
# Further Data Analysis

# Does team playing home have high changes to win?
overall = matches_df[['home_team_goal','away_team_goal']].describe()
# It is seen from the mean that home teams generally score higher than away team, in turn win the match
# Distribution of game outcomes
plt.figure(figsize=(6,6))
matches_df["home_status"].value_counts().plot.pie(autopct = "%1.0f%%",colors =sns.color_palette("rainbow",3),wedgeprops = {"linewidth":2,"edgecolor":"white"})
my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.title("DISTRIBUTION OF GAME OUTCOMES")

# Group matches by goal difference
matches_by_goal_diff = matches_df.groupby(['goal_difference']).count()['id']
matches_by_goal_diff.plot(kind='bar')
diff_by_one = matches_by_goal_diff[0]+matches_by_goal_diff[-1]+matches_by_goal_diff[1]
print (round(100*(diff_by_one)/matches_df.shape[0], 2), " % of matches is decided by one goal. ")
print (round(100*matches_by_goal_diff[0]/matches_df.shape[0], 2), " % matches were draws. " )

# Comparing a team's mean goals when played at home/away
def outcome(df):
    if df.home_team_goal > df.away_team_goal:
        return 'Home'
    elif df.home_team_goal < df.away_team_goal:
        return 'Away'
    elif df.home_team_goal == df.away_team_goal:
        return 'Draw'
# calculating the mean goals a team scored when it played home and away    
team_mean_home_scores = matches_df.groupby(['home_team_api_id'])['home_team_goal'].mean()
team_mean_away_scores = matches_df.groupby(['away_team_api_id'])['away_team_goal'].mean()
# merging both dataset based on team id
team_scores_home_away = pd.concat([team_mean_home_scores,team_mean_away_scores], axis=1, join='inner')
    
team_scores_home_away['score_higher_at'] = team_scores_home_away.apply(outcome, axis=1)
team_scores_home_away.score_higher_at.value_counts(normalize=True).plot(kind='pie')

team_scores_home_away.score_higher_at.value_counts(normalize=True).round(4)*100

team_scores_home_away[['home_team_goal','away_team_goal']].corr()


# Top teams by their home & away goals
h_t = matches_df.groupby("home_team_name")["home_team_goal"].sum().reset_index()
a_t = matches_df.groupby("away_team_name")["away_team_goal"].sum().reset_index()
h_t = h_t.sort_values(by="home_team_goal",ascending= False)
a_t = a_t.sort_values(by="away_team_goal",ascending= False)
plt.figure(figsize=(13,8))
plt.subplot(121)
ax = sns.barplot(y="home_team_name",x="home_team_goal",data=h_t[:20],palette="summer")
plt.ylabel('')
plt.title("top teams by home goals")
for i,j in enumerate(h_t["home_team_goal"][:20]):
    ax.text(.7,i,j,weight = "bold")
plt.subplot(122)
ax = sns.barplot(y="away_team_name",x="away_team_goal",data=a_t[:20],palette="winter")
plt.ylabel("")
plt.subplots_adjust(wspace = .4)
plt.title("top teams by away goals")
for i,j in enumerate(a_t["away_team_goal"][:20]):
    ax.text(.7,i,j,weight = "bold")

# Top teams by overall win
def win(data):
    if data["home_team_goal"] > data["away_team_goal"]:
        return data["home_team_name"]
    elif data["home_team_goal"] < data["away_team_goal"]:
        return data["away_team_name"]
    else:
        return "DRAW"
    
def lose(data):
    if data["home_team_goal"] < data["away_team_goal"]:
        return data["home_team_name"]
    elif data["away_team_goal"] < data["home_team_goal"]:
        return data["away_team_name"]
    elif data["home_team_goal"] == data["away_team_goal"]:
        return "DRAW"
 
matches_df['winner']  = matches_df[['home_team_goal', 'away_team_goal', 
                                    'home_team_name', 'away_team_name']].apply(win, axis=1)
matches_df["loser"] = matches_df[['home_team_goal', 'away_team_goal', 
                                    'home_team_name', 'away_team_name']].apply(lose, axis=1)

win = matches_df["winner"].value_counts()[1:].reset_index()
lost = matches_df["loser"].value_counts()[1:].reset_index()
plt.figure(figsize=(13,14))
plt.subplot(121)
ax = sns.barplot(win["winner"][:30],win["index"][:30],palette="magma")
plt.title(" TOP WINNING TEAMS")
for i,j in enumerate(win["winner"][:30]):
    ax.text(.7,i,j,color = "white",weight = "bold")
plt.subplot(122)
ax = sns.barplot(lost["loser"][:30],lost["index"][:30],palette="jet_r")
plt.title(" TOP LOSING TEAMS")
plt.subplots_adjust(wspace = .3)
for i,j in enumerate(lost["loser"][:30]):
    ax.text(.7,i,j,color = "black",weight = "bold")
    
#*************************************************************************#
# Prediction phase
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE

# Removing useless columns to process classification
df_final = matches_df.copy()
columns_to_drop = ["id", "match_api_id","date", "away_team_name","home_team_name",
                   "home_status", "home_team_goal","away_team_goal", 
                   "home_team_api_id", "away_team_api_id", 
                    "goal_difference"]

y = df_final["home_status"]
df = df_final.drop(columns_to_drop, axis=1)
df = df.fillna(0)

# Split X and y into a train and test set
X_train, X_test, y_train, y_test = train_test_split(df, y, shuffle=True, random_state=42)

# Select features using RFE
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
estimator = clf
selector = RFE(estimator, n_features_to_select = 10, step=1)
selector = selector.fit(X_train, y_train)
test = X_train.iloc[:, selector.support_].tail()
clf.fit(selector.transform(X_train), y_train)

# Calculate accuracy(overall)
score = clf.score(selector.transform(X_test), y_test)
y_pred = clf.predict(selector.transform(X_test))
score 

# Plotting results for each categorical output
import itertools
from sklearn.metrics import confusion_matrix

# Prints and plots the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = y.unique()

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')



