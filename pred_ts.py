# blue prints for models Arima ,AutoReg

"""Usage:
            from pred import Model
            df=pd.series()
            
            x=Model(data=df,freq='d',returns=False)
            x.eda()
            x.arima_mod_grid(p_param=range(0, 25, 8), q_param=range(0, 3, 1),cut=0.8)
            model=x.arima(order=(8,0,0),wfv=True) ==> Dont for git to assign it to Variable

            Returns:
                the Mae for the test set and the plot of the predictions and the residuals
"""


import pandas as pd
import numpy as np


import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from tqdm import tqdm


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

pp = pprint.PrettyPrinter(indent=1)


class Model():

    def __init__(self, data, freq, returns=False):
        """Init the class for the model object to perform the predictions

        Args:
            data (pd.Series): 
            freq (str): d, h, m, s
            returns (bool, optional): perform the predictions on returns or on close price. Defaults to False.
        Example:
            df=pd.series()
            x=Model(data=df,freq='d',returns=False)
        """
        self.freq = freq
        self.__Og_data = data.squeeze()
        self.returns = returns

    def __wrangle(self):

        data = self.__Og_data.copy()
        data.index.freq = self.freq
        returns = self.returns

        if returns:
            data = data.pct_change().dropna()
            self.__data = data
            return self.__data
        else:
            self.__data = data
            return self.__data

    def eda(self):
        """perform eda on the series
        Returns the Pacd and the Acf plots
        Args:
            None 
        example:
            Model.eda()
        """

        data = self.__wrangle()

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(25, 10))
        plot_acf(data, ax=ax1)
        ax1.set(title='Autocorrelation Function')
        plot_pacf(data, ax=ax2)
        ax2.set(title='Partial Autocorrelation Function')

    def __train_test_split(self, cut=0.8):

        data = self.__data.copy()

        y_train = data[:int(len(data)*cut)]
        y_test = data[int(len(data)*cut):]

        self.__y_train = y_train
        self.__y_test = y_test

    def arima_mod_grid(self, p_param=range(0, 25, 8), q_param=range(0, 3, 1), cut=0.80):
        """perform grid search for the best parameters of the model
           and plot the results for the grid search
           Returns:
            the parameters and the mae for each combination
        Args:
            p_param (range, optional):  Defaults to range(0, 25, 8).
            q_param (range, optional): Defaults to range(0, 3, 1).
            cut (float, optional): cutoff for the series. Defaults to 0.80.
        example:
            Model.arima_mod_grid(p_param=range(0, 25, 8), q_param=range(0, 3, 1),cut=0.8)

        Hint: 
            Q param Excluded because it takes a lot of time to compute and this is just a demo!!!!!
        """

        p_params = p_param
        q_params = q_param

        self.cut = cut

        self.__train_test_split(cut=self.cut)

        grid_dict = {}
        history = self.__y_train.copy().squeeze()
        y_train = self.__y_train.copy().squeeze()
        for p in tqdm(p_params, colour='green'):
            grid_dict[p] = []
            for q in tqdm(q_params, colour='yellow'):
                order = (p, 0, q)
                # train the model
                model = ARIMA(history, order=order).fit()
                y_pred = model.predict()
                mae = mean_absolute_error(y_train, y_pred)
                # add mae to dict
                grid_dict[p].append(mae)

        meandf = pd.DataFrame(grid_dict)
        # min in the dict
        fig = px.bar(meandf, orientation="v")
        fig.update_layout(title="Grid Search for the best parameters",
                          yaxis_title="MAE", xaxis_title="Parameters")
        fig.show()
        pp.pprint(grid_dict)

    def arima(self, order, wfv=True):
        """train the model and perform walk forward validation

            Returns trained ARIMA model Should be assigned to a variable!!!!!!!
            Test mae and the plot of the predictions , residuals and diagnostics plots.

        Args:
            order (tuple): the parameters of the model ==> order = (p, 0, q)
            wfv (bool, optional): perform walk forward val.. Defaults to True.

        example:
            model=Model.arima(order=(8,0,0),wfv=True) ==> Dont for git to assign it to Variable

        Hint: 
            Q param Excluded because it takes a lot of time to compute and this is just a demo!!!!!

        """

        self.__train_test_split(cut=self.cut)

        if wfv:
            y_test = self.__y_test.copy().squeeze()
            y_pred_wfv = pd.Series()
            history = self.__y_train.copy().squeeze()
            for i in tqdm(range(len(y_test)), colour='green'):
                model = ARIMA(history, order=order).fit()
                next_pred = model.forecast()
                y_pred_wfv = y_pred_wfv.append(next_pred)
                history = history.append(y_test[next_pred.index])

            test_mae = mean_absolute_error(y_test, y_pred_wfv)
            print(f"Test MAE===>{test_mae} ")

            # plot predictions

            df_pred = pd.DataFrame({"y_test": y_test, "y_pred": y_pred_wfv})
            fig = px.line(df_pred)
            fig.update_layout(title="Predictions Vs True Values",
                              xaxis_title="Date", yaxis_title="Price or Returns")
            fig.show()

            # plot residuals
            df_resid = pd.DataFrame({"residuals": y_test-y_pred_wfv})
            fig = px.line(df_resid)
            fig.update_layout(title="Residuals", xaxis_title="Date",
                              yaxis_title="Price or Returns")
            fig.show()

            # plot diagnostics
            fig, ax = plt.subplots(figsize=(15, 12))
            model.plot_diagnostics(fig=fig)
            # print summary
            print(model.summary())

            return model

        else:
            cut = cut
            self.__train_test_split(cut=cut)

            order = order
            history = self.__y_train.copy().squeeze()
            y_train = self.__y_train.copy().squeeze()
            model = ARIMA(history, order=order).fit()
            y_pred = model.predict()
            mae = mean_absolute_error(y_train, y_pred)
            print(mae)

            fig, ax = plt.subplots(figsize=(15, 12))
            model.plot_diagnostics(fig=fig)

            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(y_train, label='Train')
            ax.plot(y_pred, label='ARIMA')
            ax.set(title='ARIMA', xlabel='Date', ylabel='Price')
            ax.legend()
            plt.show()
            print(model.summary())
            return model
