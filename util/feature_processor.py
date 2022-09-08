from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd
from util.index import log


def select_features(df: pd.DataFrame, num: int):
    df = df.copy()
    y = df.pop('label')
    x = df
    fit = SelectKBest(score_func=chi2, k=len(x.columns)).fit(x, y)
    df_scores = pd.DataFrame(fit.scores_, columns=['score'])
    df_columns = pd.DataFrame(x.columns, columns=['feature_name'])
    df_result = pd.concat([df_scores, df_columns], axis=1).sort_values(by="score", ascending=False, ignore_index=True)
    selected_features = df_result.loc[0:num-1]
    log('selected features as follows:')
    print(selected_features)
    return list(selected_features.loc[:, 'feature_name'])
