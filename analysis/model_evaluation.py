from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, make_scorer
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
        scoring={
            "neg_mean_absolute_error":make_scorer(lambda ytrue, ypred: -mean_absolute_error(ytrue, ypred)),
            "neg_root_mean_squared_error":make_scorer(lambda ytrue, ypred: -root_mean_squared_error(ytrue, ypred)),
            "diff": make_scorer(lambda ytrue, ypred: ytrue - ypred)
            },
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
    diff = cv_results['test_diff']
    print(
        f"Mean Absolute Error:     {mae.mean():.5f} +/- {mae.std():.5f}\n"
        f"Median MAE + [2.5, 97.5] Percentiles: {np.median(mae):.5f} + [{np.percentile(mae, 2.5):.5f}, {np.percentile(mae, 97.5):.5f}]\n"
        f"Root Mean Squared Error: {rmse.mean():.5f} +/- {rmse.std():.5f}"
    )

    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(25, 10))
        
        # Plot MAE distribution
    
        pd.Series(mae).hist(ax=axes[0], density=True, alpha=1, label='MAE', bins=15)
        axes[0].set_title('Absolute Error Distribution')
        axes[0].set_xlabel('$|y_{true} - y_{pred}|$', fontsize = 20)
        axes[0].legend()

        # Plot RMSE distribution
        # pd.Series(rmse).hist(ax=axes[0, 1], density=True, alpha=0.7, label='RMSE', bins=15)
        # axes[0, 1].set_title('Root Mean Squared Error Distribution')
        # axes[0, 1].set_xlabel('RMSE')
        # axes[0, 1].legend()

        # # Plot MAE error bar
        # # y.plot(ax = axes[1,0])
        # axes[1, 0].errorbar(x=y.iloc[-len(mae):].index, y=y.values[-len(mae):], yerr=mae/2, fmt='o', color='blue', capsize=5)
        # axes[1, 0].set_title('Mean Absolute Error with Error Bar')

        # Plot RMSE error bar
        # y.plot(ax = axes[1,1])
        axes[1].plot(y.index, y.values)
        # axes[1, 1].errorbar(x=y.iloc[-len(diff):].index, y=y.values[-len(diff):], yerr=diff, fmt='o', color='green', capsize=5)
        axes[1].plot(y.iloc[-len(diff):].index, y.values[-len(diff):]+diff, linestyle = '--', color = 'black')
        colors = ['green' if d > 0 else 'red' for d in diff]
        axes[1].vlines(
            x=y.iloc[-len(diff):].index,
            ymin=y.values[-len(diff):],
            ymax=y.values[-len(diff):]+diff,
            colors = colors, alpha = 0.5
            )
        axes[1].tick_params(axis='x', labelrotation=90)
        axes[1].set_title('Ground truth vs. prediction')
        axes[1].set_ylabel('ILI incidence / 10^5')
        axes[1].legend(['ground truth', 'predictions'])
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        import seaborn as sns
        # Create a DataFrame for Seaborn
        df = pd.DataFrame({'Ground Truth': y.values[-len(diff):], 'Prediction': y.values[-len(diff):] + diff})
        # Use lmplot to plot predicted vs actual values
        sns.regplot(x='Ground Truth', y='Prediction', data=df, ax=axes[2])
        axes[2].set_xlabel('Ground Truth')
        axes[2].set_ylabel('Prediction')
        axes[2].set_title('Prediction vs Ground Truth')

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
    fig, ax = plt.subplots(figsize = (25,25))
    
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