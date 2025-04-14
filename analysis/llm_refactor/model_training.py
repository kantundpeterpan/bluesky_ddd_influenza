import joblib
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from typing import Optional, List
from analysis.model_evaluation import evaluate, plot_predictions, plot_feature_importance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(Xtrain: pd.DataFrame, ytrain: pd.Series, model_type: str = 'HistGradientBoostingRegressor', restrict_model: bool = False):
    """Trains a time series model.

    Args:
        Xtrain (pd.DataFrame): Training features.
        ytrain (pd.Series): Training target.
        model_type (str): Type of model to use ('HistGradientBoostingRegressor' or 'Ridge').
        restrict_model (bool): Whether to restrict model parameters.

    Returns:
        The fitted model.
    """
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
            else:
                model = HistGradientBoostingRegressor(random_state=42)
            logging.info("Using HistGradientBoostingRegressor model.")
        elif model_type == 'Ridge':
            model = Ridge()
            logging.info("Using Ridge model.")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(Xtrain, ytrain)
        logging.info("Model fitted successfully.")
        print()
        print(f"Training R^2: {model.score(Xtrain, ytrain):.3f}")
        print(f"Training MAE: {mean_absolute_error(ytrain, model.predict(Xtrain)):.5f}")
        print(f"Training RMSE: {root_mean_squared_error(ytrain, model.predict(Xtrain)):.5f}")
        print()

        return model

    except ValueError as e:
        logging.error(str(e))
        raise
    except Exception as e:
        logging.error(f"Error fitting model: {e}")
        raise

def prepare_data(
    df: pd.DataFrame, lags: int = 2, weeks_ahead: int = 1,
    target_col: str = 'ili_incidence', normalize_y: bool = True,
    cols_to_drop: Optional[List[str]] = None, split_data: bool = False,
    split_date: str = '2024-08-01'
) -> tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], Optional[pd.Series]]:
    """Prepares data for model training and evaluation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lags (int): Number of lag weeks to use as features.
        weeks_ahead (int): Number of weeks ahead to predict.
        target_col (str): Name of the target column.
        normalize_y (bool): Whether to normalize the target variable.
        cols_to_drop (list): List of columns to drop from the feature set.
        split_data (bool): Whether to split the data into training and testing sets.
        split_date (str): Date to split the data into training and testing sets.

    Returns:
        Tuple containing Xtrain, ytrain, Xtest (optional), and ytest (optional).
    """
    try:
        from analysis.llm_refactor.data_processing import make_train_test, prepare_data_cv # Import here to avoid circular dependency

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
        logging.info(Xtrain.shape)
        
        return Xtrain, ytrain, Xtest, ytest
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        raise

def evaluate_model(model, Xtrain, ytrain, Xtest, ytest, model_type, figure_path, dataset, split_data, df, lang, target_col):
    """Evaluates the trained model and generates plots.

    Args:
        model: The trained model.
        Xtrain (pd.DataFrame): Training features.
        ytrain (pd.Series): Training target.
        Xtest (pd.DataFrame): Testing features.
        ytest (pd.Series): Testing target.
        model_type (str): Type of model used.
        figure_path (str): Path to save the generated figures.
        dataset (str): Name of the dataset used.
        split_data (bool): Whether the data was split into training and testing sets.
        df (pd.DataFrame): The input DataFrame.
        lang (str): The language.
        target_col (str): Name of the target column.
    """
    try:
        # Define TimeSeriesSplit
        cv =  TimeSeriesSplit(
            n_splits=42 if 'upsampled' not in dataset else 220,
            gap=0,
            max_train_size=200,
            test_size=1,
        )
        
        # Evaluate model
        if model_type == 'HistGradientBoostingRegressor':
            fig, cv_res = evaluate(model, Xtrain, ytrain, cv, plot = True, return_fig=True, return_cv_res=True)
            Path(figure_path).mkdir(parents=True, exist_ok=True)
            fig.savefig(os.path.join(figure_path, 'evaluations.png'))
        
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

        from analysis.llm_refactor.data_processing import save_results  # Import here to avoid circular dependency
        save_results(df, lang, target_col, model_type, dataset, cv_res, imp_df)

    except Exception as e:
        logging.error(f"Error evaluating the model: {e}")
        raise

def save_model(model, output_path: str):
    """Saves the trained model to disk.

    Args:
        model: The trained model.
        output_path (str): Path to save the fitted model.
    """
    logging.info(f"Saving model to {output_path}")
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logging.info(f"Model saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise


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

    from analysis.llm_refactor.google_cloud import create_service_account_credentials  # Import here to avoid circular dependency

    credentials = None
    if gc_creds_env:
        credentials = create_service_account_credentials()


    # Load data
    try:
        from analysis.llm_refactor.data_loading import load_data # Import here to avoid circular dependency
        
        df = load_data(data_path, dataset, credentials)
        
        lang = dataset.split("_")[-1]

        logging.info("Data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

    # Prepare data
    Xtrain, ytrain, Xtest, ytest = prepare_data(
        df, lags=lags, weeks_ahead=weeks_ahead,
        target_col=target_col, normalize_y=normalize_y,
        cols_to_drop=cols_to_drop, split_data=split_data,
        split_date=split_date
    )

    # Choose model
    from analysis.llm_refactor.data_processing import create_model # Import here to avoid circular dependency
    model = create_model(model_type, restrict_model)

    evaluate_model(model, Xtrain, ytrain, Xtest, ytest, model_type, figure_path, dataset, split_data, df, lang, target_col)

    # Save model
    save_model(model, output_path)
