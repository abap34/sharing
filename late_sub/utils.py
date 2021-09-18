import datetime
import tqdm
import numpy as np
import pandas as pd


def read_data(data_contain_path):
    train = pd.read_csv(data_contain_path + '/train.csv')
    test = pd.read_csv(data_contain_path + '/test.csv')
    train["is_train"] = True
    test["is_train"] = False
    submit = pd.read_csv(data_contain_path + '/sample_submission.csv')
    return train, test, submit


def read_wave(paths):
    X = []
    for p in tqdm.tqdm(paths):
        X.append(np.load(p))
    return np.stack(X)



def info(*messages):
    m = ""
    for massage in messages:
        m += str(massage)
    dt_now = datetime.datetime.now()
    t = dt_now.strftime('[%H:%M:%S]')
    print('[info] {} {}'.format(t, m))


