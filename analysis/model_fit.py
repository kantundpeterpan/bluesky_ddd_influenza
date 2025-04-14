import sys
import os
import argparse
import joblib
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("../"))

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import cross_validate
from analysis.load_dfs import (
    load_merged_posts_ww, make_train_test,
    load_post_count_ili, prepare_data_cv,
    load_llm_filtered_post_count,
    load_post_count_ili_upsampled
)
from analysis.model_evaluation import evaluate, plot_predictions, plot_feature_importance
from analysis.gc_tools import create_service_account_credentials

from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")
plt.rcParams.update({
    'figure.figsize': (8, 6),  # Adjust figure size for SVG
    'figure.dpi': 250,       # Set DPI for SVG output
    'svg.fonttype': 'none',   # Embed fonts in SVG for portability
    'font.size': 14,
    'axes.titlesize': 24,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'legend.facecolor': 'white',
    'legend.framealpha':1
})

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# fit HistGradient with post time series and lag features

def fit_and_evaluate(
    data_path: str,
    dataset: str = 'grippe_posts_fr',
    model_type: str = 'HistGradientBoostingRegressor',
    split_date: str = '2024-08-01', lags: int = 2, weeks_ahead: int = 1,
    target_col: str = 'ili_incidence', normalize_y: bool = True,
    cols_to_drop: list = None, split_data: bool = False,
    output_path: str = 'model.joblib',
    figure_path: str = 'figures',
    restrict_model: bool = False,
    gc_creds_env: bool = False
):
    """
    Loads data, preprocesses it, fits a model, evaluates it using TimeSeriesSplit,
    and saves the fitted model to disk.

    Args:
        data_path (str): Path to the data (either a csv or instructs to load data).
        dataset (str): Name of the dataset to load ('grippe_posts' or 'llm_filtered').
        model_type (str): Type of model to use ('HistGradientBoostingRegressor' or 'Ridge').
        split_date (str): Date to split the data into training and testing sets.
        lags (int): Number of lag weeks to use as features.
        weeks_ahead (int): Number of weeks ahead to predict.
        target_col (str): Name of the target column.
        normalize_y (bool): Whether to normalize the target variable.
        cols_to_drop (list): List of columns to drop from the feature set.
        output_path (str): Path to save the fitted model.
        figure_path (str): Path to save figures.
        restrict_model (bool): Whether to restrict model parameters.
    """

    logging.info(f"Starting fit_and_evaluate with data_path={data_path}, dataset={dataset}, model_type={model_type}, split_date={split_date}, lags={lags}, weeks_ahead={weeks_ahead}, target_col={target_col}, normalize_y={normalize_y}, cols_to_drop={cols_to_drop}, output_path={output_path}, figure_path={figure_path}, split_data={split_data}, restrict_model={restrict_model}")

    credentials = None
    if gc_creds_env:
        credentials = create_service_account_credentials()


    # Load data
    try:
       # if os.path.exists(data_path):
       #     df = pd.read_csv(data_path, index_col='date', parse_dates=['date'])
        #else:
        if dataset.startswith("grippe_posts"):
            lang = dataset.split("_")[-1]
            df = load_post_count_ili(lang, credentials=credentials)
            df = df.rename(columns={'rest_posts':'control_posts', 'grippe_posts':'ili_posts'})
        
        elif dataset == 'llm_filtered':
            df = load_llm_filtered_post_count(credentials=credentials)
            
        elif 'upsampled' in dataset:
            df = load_post_count_ili_upsampled(dataset.split("_")[-1], credentials=credentials).loc['2023-08-06':'2025-03-30']
        
        else:
            raise ValueError("Unsupported dataset")
        
        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Prepare data
    try:
        if split_data:
            Xtrain, ytrain, Xtest, ytest = make_train_test(
            df, split_date=split_date, lags=lags, weeks_ahead=weeks_ahead,
            target_col=target_col, normalize_y=normalize_y, cols_to_drop=cols_to_drop
            )
        else:
            Xtest, ytest = None, None
            Xtrain, ytrain = prepare_data_cv(
                df, lags=lags, weeks_ahead=weeks_ahead,
                target_col=target_col, normalize_y=normalize_y, cols_to_drop=cols_to_drop
            )
        logging.info("Data prepared for training and testing.")

        Xtrain = Xtrain.loc[~ytrain.isna().values]
        ytrain = ytrain.dropna()
        
        if Xtest is not None:
            Xtest = Xtest.loc[~ytest.isna().values]
            ytest = ytest.dropna()
            
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

    # Choose model
    try:
        if model_type == 'HistGradientBoostingRegressor':
            if restrict_model:
                model = HistGradientBoostingRegressor(
                            max_leaf_nodes=10,
                            max_depth=3,
                            min_samples_leaf=10,
                            learning_rate=0.05,
                            max_iter=100,
                            l2_regularization=1.0,
                            early_stopping='auto',
                            validation_fraction=0.2,
                            max_features=0.8,
                            random_state=42
                        )
                # model = HistGradientBoostingRegressor(
                #         max_leaf_nodes=10,          # Limit the number of leaves per tree (simpler trees)
                #         max_depth=3,                # Restrict the depth of trees to prevent overfitting
                #         min_samples_leaf=10,        # Ensure each leaf has at least 10 samples
                #         learning_rate=0.025,         # Use a low learning rate for gradual optimization
                #         max_iter=100,               # Restrict the number of boosting iterations
                #         l2_regularization=1.0,      # Add strong L2 regularization to penalize complexity
                #         early_stopping='auto',      # Stop training early if validation loss stops improving
                #         validation_fraction=0.2,    # Use 20% of data for validation during training
                #         max_features=0.8,        # Randomly sample features for splits (reduces overfitting)
                #         random_state=42             # Set random seed for reproducibility
                #     )
            else:
                model = HistGradientBoostingRegressor(random_state=42)
            logging.info("Using HistGradientBoostingRegressor model.")
        elif model_type == 'Ridge':
            model = Ridge()
            logging.info("Using Ridge model.")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    except ValueError as e:
        logging.error(str(e))
        raise

    # Define TimeSeriesSplit
    cv =  TimeSeriesSplit(
        n_splits=42 if 'upsampled' not in dataset else 220,
        gap=0,
        max_train_size=200,
        test_size=1,
    )
    
    # for i, (train_index, test_index) in enumerate(cv.split(Xtrain)):

    #     print(f"Fold {i}:")

    #     print(f"  Train: index={train_index}")

    #     print(f"  Test:  index={test_index}")

    # Evaluate model
    if model_type == 'HistGradientBoostingRegressor':
        fig, cv_res = evaluate(model, Xtrain, ytrain, cv, plot = True, return_fig=True, return_cv_res=True)
        Path(figure_path).mkdir(parents=True, exist_ok=True)
        fig.savefig(os.path.join(figure_path, 'evaluations.png'))
        

    # Fit model
    try:
        model.fit(Xtrain, ytrain)
        logging.info("Model fitted successfully.")
        print()
        print(f"Training R^2: {model.score(Xtrain, ytrain):.3f}")
        print(f"Training MAE: {mean_absolute_error(ytrain, model.predict(Xtrain)):.5f}")
        print(f"Training RMSE: {root_mean_squared_error(ytrain, model.predict(Xtrain)):.5f}")
        print()
        if split_data:
            print(f"Test R^2: {model.score(Xtest, ytest):.3f}")
            print(f"Test MAE: {mean_absolute_error(ytest, model.predict(Xtest)):.5f}")
            print(f"Test RMSE: {root_mean_squared_error(ytest, model.predict(Xtest)):.5f}")
            print()
    except Exception as e:
        logging.error(f"Error fitting model: {e}")
        raise


    fontsize = 20
    yoffset_fontsize= 15
    fontsize_legend = 15
    linewidth = 4
    # Make plots for test set
    try:
        fig1 = plot_predictions(model,
                                Xtrain, ytrain,
                                Xtest=Xtest if Xtest is not None else None,
                                ytest=ytest if ytest is not None else None,
                            is_fitted = True,
                            return_fig=True)
        
        
        # Apply formatting to the figure
        for ax in fig1.get_axes():
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
            if ax.title:
                ax.set_title(ax.get_title(), fontsize=fontsize, fontweight = 'bold')

            # Adjust legend fontsize if a legend exists
            if ax.get_legend():
                for text in ax.get_legend().get_texts():
                    text.set_fontsize(fontsize_legend)
        fig1.tight_layout()
        
        Path(figure_path).mkdir(parents=True, exist_ok=True)
        fig1.savefig(os.path.join(figure_path, 'predictions.png'))
        logging.info(f"Predictions plot saved to {os.path.join(figure_path, 'predictions.png')}")
        
        fig2, ax, imp_df = plot_feature_importance(
            model,
            Xtrain if Xtest is None else Xtest,
            ytrain if ytest is None else ytest , 
            n_repeats=25, random_state=42, n_jobs=-1,
            keep_n=10
        )

        fig2.tight_layout()

        if fig2 is None:
            logging.warning("Feature importance plot returned None.")
        else:
            fig2.savefig(os.path.join(figure_path, 'feature_importance.png'))
            logging.info(f"Feature importance plot saved to {os.path.join(figure_path, 'feature_importance.png')}")

    except Exception as e:
        logging.error(f"Error creating plots: {e}")
        raise

    # Save model
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logging.info(f"Model saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

    # save results for dashboarding
    res = pd.DataFrame()
    res['date'] = df.index
    res['lang'] = lang
    res['target'] = df[target_col].values*100_000
    print(df[target_col])
    res['cv_preds'] = pd.NA
    
    no_val_points = len(cv_res['test_diff']) 
    res.iloc[-no_val_points:, -1][:] = res['target'].iloc[-no_val_points:].values + cv_res['test_diff']
    res['abs_diff'] = pd.NA
    res.iloc[-no_val_points:, -1] = np.abs(cv_res['test_diff'])

    res.to_csv(f"../dbt/digepi_bsky/seeds/{model_type}_{dataset}.csv", index=False)
    imp_df.melt().to_csv(f"../dbt/digepi_bsky/seeds/{model_type}_{dataset}_featureimp.csv", index = False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit and evaluate a time series model.")
    parser.add_argument("--data_path", type=str, default=".",
                        help="Path to the data file (or data loading instruction).")
    parser.add_argument("--dataset", type=str, default="grippe_posts_fr", choices=['grippe_posts_fr', 'grippe_posts_de', 'llm_filtered', 'upsampled_de', 'upsampled_fr'],
                        help="Name of the dataset to load ('grippe_posts' or 'llm_filtered').")
    parser.add_argument("--model_type", type=str, default="HistGradientBoostingRegressor",
                        help="Type of model to use (HistGradientBoostingRegressor or Ridge).")
    parser.add_argument("--split_date", type=str, default="2024-08-01",
                        help="Date to split the data into training and testing sets.")
    parser.add_argument("--lags", type=int, default=2,
                        help="Number of lag weeks to use as features.")
    parser.add_argument("--weeks_ahead", type=int, default=1,
                        help="Number of weeks ahead to predict.")
    parser.add_argument("--target_col", type=str, default="ili_incidence",
                        help="Name of the target column.")
    parser.add_argument("--normalize_y", action="store_true",
                        help="Whether to normalize the target variable.")  # Use store_true for boolean
    parser.add_argument("--cols_to_drop", nargs='+', type=str, default=['ili_case', 'ari_case', 'ili_incidence',
                                                                        'ari_incidence', 'ili_pop_cov', 'ari_pop_cov'],
                        help="List of columns to drop from the feature set.")
    parser.add_argument("--output_path", type=str, default="model.joblib",
                        help="Path to save the fitted model.")
    parser.add_argument("--figure_path", type=str, default="figures",
                        help="Path to save the generated figures.")
    parser.add_argument("--split_data", action="store_true",
                        help="Whether to split the data into training and testing sets.")
    parser.add_argument("--restrict_model", action="store_true",
                        help="Whether to restrict the model parameters")
    parser.add_argument("--gc_creds_env", action="store_true",
                        help="Whether to read Google Cloud credentials from environment variables")
    args = parser.parse_args()

    fit_and_evaluate(
    data_path=args.data_path,
    dataset=args.dataset,
    model_type=args.model_type,
    split_date=args.split_date,
    lags=args.lags,
    weeks_ahead=args.weeks_ahead,
    target_col=args.target_col,
    normalize_y=args.normalize_y,
    cols_to_drop=args.cols_to_drop,
    output_path=args.output_path,
    figure_path=args.figure_path,
    split_data=args.split_data,
    restrict_model=args.restrict_model,
    gc_creds_env=args.gc_creds_env
    )
