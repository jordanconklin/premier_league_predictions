import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_and_prepare_data():
    # Load the data
    data = pd.read_csv('data.csv')

    # Define the features 
    # TODO adjust these features
    features = [
        'home_team_rank',
        'away_team_rank',
        'home_team_goals_scored_avg',
        'away_team_goals_scored_avg',
        'home_team_goals_conceded_avg',
        'away_team_goals_conceded_avg'
    ]