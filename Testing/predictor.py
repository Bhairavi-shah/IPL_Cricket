### Custom definitions and classes if any ###
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
def predictRuns(testInput):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(dir_path+"/"+testInput)
    

    # Batsmen & Bowler ecoding
    file = open(dir_path+"/batsmen.pkl", "rb")
    batsmen_dict = pickle.load(file)
    file = open(dir_path+"/bowlers.pkl", "rb")
    bowlers_dict = pickle.load(file)

    for i, row in df.iterrows():
        bats = []
        for bat in row['batsmen'].split(","):
            bat = bat.strip()
            if bat in batsmen_dict.keys():
                bats.append(batsmen_dict[bat])
            else:
                batsmen_dict[bat] = len(batsmen_dict)
                bats.append(len(batsmen_dict))
        df.at[i, "batsmen"] = bats

        bats = []
        for bat in row['bowlers'].split(","):
            bat = bat.strip()
            if bat in bowlers_dict.keys():
                bats.append(bowlers_dict[bat])
            else:
                batsmen_dict[bat] = len(bowlers_dict)
                bats.append(len(bowlers_dict))
        df.at[i, "bowlers"] = bats


    df = pd.concat([df.reset_index(), pd.DataFrame(df.batsmen.values.tolist()).add_prefix('batsmen_')],axis=1)
    df = pd.concat([df.reset_index(), pd.DataFrame(df.bowlers.values.tolist()).add_prefix('bowler_')],axis=1)
    df = df.drop('batsmen', axis = 1)
    df = df.drop('bowlers', axis = 1)

    venue_encoder = LabelEncoder()
    team_encoder = LabelEncoder()
    venue_encoder.classes_ = np.load(dir_path+'/venue_encoder.npy', allow_pickle=True)
    team_encoder.classes_ = np.load(dir_path+'/team_encoder.npy', allow_pickle=True)

    df['venue'] = venue_encoder.transform(df['venue'])
    df['batting_team'] = team_encoder.transform(df['batting_team'])
    df['bowling_team'] = team_encoder.transform(df['bowling_team'])

    cols = ['index', 'venue', 'innings', 'batting_team', 'bowling_team',
       'batsmen_0', 'batsmen_1', 'batsmen_2', 'batsmen_3', 'batsmen_4',
       'batsmen_5', 'batsmen_6', 'batsmen_7', 'batsmen_8', 'batsmen_9',
       'batsmen_10', 'batsmen_11', 'batsmen_12', 'bowler_0', 'bowler_1',
       'bowler_2', 'bowler_3', 'bowler_4', 'bowler_5', 'bowler_6', 'bowler_7',
       'bowler_8', 'bowler_9']
    df = df.drop("level_0", axis = 1)
    df = df.reindex(df.columns.union(cols, sort=False), axis=1, fill_value=0)

    sc = pickle.load(open(dir_path+'/scaler.pkl', 'rb'))
    x_test = sc.transform(df)

    new_model = tf.keras.models.load_model('SequentialModel')
    y_pred2 = new_model.predict(x_test)
    target_scaler = pickle.load(open(dir_path+'/target_scaler.pkl', 'rb'))
    y_pred2 = target_scaler.inverse_transform(y_pred2)

    return int(y_pred2[0][0].round())
