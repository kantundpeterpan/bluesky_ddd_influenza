from sklearn.preprocessing import SplineTransformer
import numpy as np

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

