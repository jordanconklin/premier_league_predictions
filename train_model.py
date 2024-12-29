import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_and_prepare_data(stat_type):
    # Load player data
    df = pd.read_csv(f'data/player_{stat_type}.csv')
    
    # Features for player prediction
    features = [
        'minutes_played_last_5',
        'average_' + stat_type + '_last_5',
        'home_or_away',  # 1 for home, 0 for away
        'opponent_rank',
        'days_rest',
        'player_form_rating',
        'team_goals_last_5',
        'opponent_goals_allowed_last_5'
    ]
    
    X = df[features]
    y = df[stat_type + '_actual'] # the actual result for the stat
    
    return X, y

