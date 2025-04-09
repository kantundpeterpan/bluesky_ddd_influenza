from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt

def evaluate(model, X, y, cv, model_prop=None, model_step=None, plot=False, plot_filepath="evaluation_metrics.png", return_fig=False):
    """
    Evaluates a model using cross-validation and optionally plots the distribution of MAE and RMSE.

    Args:
        model: The scikit-learn model to evaluate.
        X: The feature data.
        y: The target data.
        cv: The cross-validation strategy (e.g., KFold, StratifiedKFold).
        model_prop: Optional. If provided, prints the mean value of this property from the fitted models.
        model_step: Optional. If provided, gets the model property from a specific step in a pipeline.
        plot: Boolean flag to enable plotting the distribution of MAE and RMSE.
        plot_filepath: Filepath to save the plot (if plot is True).
        return_fig: Boolean flag to return the matplotlib figure object.

    Returns:
        None.  Prints evaluation metrics, and optionally saves/returns a plot.
    """
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        return_estimator=model_prop is not None,
    )
    if model_prop is not None:
        if model_step is not None:
            values = [
                getattr(m[model_step], model_prop) for m in cv_results["estimator"]
            ]
        else:
            values = [getattr(m, model_prop) for m in cv_results["estimator"]]
        print(f"Mean model.{model_prop} = {np.mean(values)}")
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.5f} +/- {mae.std():.5f}\n"
        f"Root Mean Squared Error: {rmse.mean():.5f} +/- {rmse.std():.5f}"
    )

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot MAE distribution
        pd.Series(mae).hist(ax=axes[0], density=True, alpha=0.7, label='MAE')
        pd.Series(mae).plot(kind='kde', ax=axes[0], color='blue')
        axes[0].set_title('Mean Absolute Error Distribution')
        axes[0].set_xlabel('MAE')
        axes[0].legend()

        # Plot RMSE distribution
        pd.Series(rmse).hist(ax=axes[1], density=True, alpha=0.7, label='RMSE')
        pd.Series(rmse).plot(kind='kde', ax=axes[1], color='green')
        axes[1].set_title('Root Mean Squared Error Distribution')
        axes[1].set_xlabel('RMSE')
        axes[1].legend()
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping


        if return_fig:
            return fig



def plot_predictions(
    predictor, Xtrain, ytrain,
    Xtest = None, ytest = None,
    is_fitted: bool = True, return_fig: bool = False,
    true_legend_str: str = 'true', pred_legend_str: str = 'predicted', 
):
    if not is_fitted:
        predictor.fit(Xtrain, ytrain)
        
    ytrainpred = pd.Series(
        predictor.predict(Xtrain), index = ytrain.index
    )
    
    fig, ax = plt.subplots(figsize = (10,10))
    
    ax = ytrain.plot(label = f"ILI incidence {true_legend_str}")   
    ytrainpred.plot(label = f"{pred_legend_str}", ax = ax)
    
    ax.set_title("True vs. predicted ILI incidence", fontweight = 'bold')
    ax.set_ylabel("weekly ILI incidence / 100 000", fontweight = 'bold')
    ax.legend()
    
    if Xtest is not None:
        ytestpred = pd.Series(
            predictor.predict(Xtest), index = ytest.index
        ) 
        ytest.plot()
        ytestpred.plot()
        
    if return_fig:
        return plt.gcf()    
    
def plot_feature_importance(
    predictor, Xtest, ytest, 
    n_repeats=25, random_state=42, n_jobs=-1,
    keep_n: int = None
):
    fig, ax = plt.subplots(figsize = (15,15))
    
    result = permutation_importance(
        predictor, Xtest, ytest, n_repeats=25, random_state=42, n_jobs=-1,
        scoring='neg_mean_absolute_error'
    )
    
    sorted_importances_idx = (result.importances_mean*-1).argsort()[::-1]
    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=Xtest.columns[sorted_importances_idx],
    )
    
    if keep_n:
        importances = importances.iloc[:,-keep_n:]
        
    ax = importances.plot.box(vert=False, whis=10)
    ax.set_title("Permutation Importances", fontweight = "bold")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Increase in MAE")
    ax.figure.tight_layout()
    
    
    return ax.figure, ax