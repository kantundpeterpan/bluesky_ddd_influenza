from sklearn.preprocessing import SplineTransformer, OneHotEncoder
import numpy as np
from typing import List

def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def extact_time_features(df):
    
    df['year'] = df.index.year.astype("category")
    df['month'] = df.index.month.astype("category")
    df['week'] = df.index.isocalendar().week.astype("category")
    df['season'] = df['month'].apply(assign_season).astype("category")
    
    return df

def onehot_encoding(
    df, cols_to_encode: List[str]
):
    ohe = OneHotEncoder(sparse_output = False)
    ohe.fit(df[cols_to_encode])
    seasonal_encoded = pd.DataFrame(
        ohe.fit_transform(df[['year', 'month', 'week', 'season']]),
        index = df.index, columns = ohe.get_feature_names_out()
    )
    df = pd.concat([
        df.drop(['year', 'month', 'week', 'season'], axis = 1),
        seasonal_encoded
    ], axis = 1)
    
    return df