
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pywaffle import Waffle
from IPython.display import Image, display

class tastyPlots:
    @staticmethod
    def _validate_dataframe(df):
        """
        Validate if the input is a pandas DataFrame or Series.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data to be validated.

        Raises:
        - ValueError: If the input is not a pandas DataFrame or Series.
        """
        if not isinstance(df, pd.DataFrame) and not isinstance(df, pd.Series):
            tastyPlots.show_fail_gif()
            print("Введи нормальні дані :)")
            raise ValueError("Input must be a pandas DataFrame or Series")

    @staticmethod
    def validate_dataframe(func):
        """
        Decorator to validate input as a pandas DataFrame or Series.

        Parameters:
        - func (function): The function to be decorated.

        Returns:
        - function: Decorated function.
        """
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, (pd.DataFrame, pd.Series)):
                    tastyPlots._validate_dataframe(arg)
            return func(*args, **kwargs)
        return wrapper

    @staticmethod
    def show_fail_gif():
        """
        Display a GIF for failure.
        """
        display(Image(url='TastyModule/tastyError.gif'))

    @staticmethod
    @validate_dataframe
    def pieChart(df, title="Default Pie Chart"):
        """
        Generate and display a pie chart.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the pie chart.
        - title (str): Title of the pie chart.
        """
        try:
            df.plot(kind='pie', subplots=True, figsize=(8, 8))
            plt.title(title)
            plt.ylabel("")
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting pie chart: {e}")
            tastyPlots.show_fail_gif()

    @staticmethod
    @validate_dataframe
    def lolipop1(df, title="Just lolipops", ylabel="its a loli y label"):
        """
        Generate and display a lolipop chart.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the lolipop chart.
        - title (str): Title of the lolipop chart.
        - ylabel (str): Y-axis label of the lolipop chart.
        """
        try:
            # Draw plot
            fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
            ax.vlines(x=df.index, ymin=0, ymax=df.iloc[:, 1], color='firebrick', alpha=0.7, linewidth=2)
            ax.scatter(x=df.index, y=df.iloc[:, 1], s=75, color='firebrick', alpha=0.7)

            # Title, Label, Ticks and Ylim
            ax.set_title(title, fontdict={'size': 22})
            ax.set_ylabel(ylabel)
            ax.set_xticks(df.index)
            ax.set_xticklabels(df.iloc[:, 0].str.upper(), rotation=60,
                               fontdict={'horizontalalignment': 'right', 'size': 12})
            ax.set_ylim(0, 30)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting lolipop1: {e}")
            tastyPlots.show_fail_gif()

    @staticmethod
    @validate_dataframe
    def horiLolipop(df, title="Just lolipops", ylabel="its a loli y label", xlabel="its a loli x label"):
        """
        Generate and display a horizontal lolipop chart.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the horizontal lolipop chart.
        - title (str): Title of the horizontal lolipop chart.
        - ylabel (str): Y-axis label of the horizontal lolipop chart.
        - xlabel (str): X-axis label of the horizontal lolipop chart.
        """
        try:
            my_range = range(1, len(df.index) + 1)
            # The horizontal plot is made using the hline function
            plt.hlines(y=my_range, xmin=0, xmax=df.iloc[:, 1], color='skyblue')
            plt.plot(df.iloc[:, 1], my_range, "o")

            # Add titles and axis names
            plt.yticks(my_range, df.iloc[:, 0])
            plt.title(title, loc='left')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            # Show the plot
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting horiLolipop: {e}")
            tastyPlots.show_fail_gif()

    @staticmethod
    @validate_dataframe
    def wafel(df):
        """
        Generate and display a waffle chart.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the waffle chart.
        """
        try:
            plt.figure(
                FigureClass=Waffle,
                rows=5,
                columns=20,
                values=df,
                legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)},
            )
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting wafel: {e}")
            tastyPlots.show_fail_gif()

    @staticmethod
    @validate_dataframe
    def donut(df):
        """
        Generate and display a donut chart.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the donut chart.
        """
        try:
            # Create a circle at the center of the plot
            my_circle = plt.Circle((0, 0), 0.7, color='white')

            # Custom wedges
            plt.pie(df.iloc[:, 1], labels=df.iloc[:, 0], wedgeprops={'linewidth': 7, 'edgecolor': 'white'})
            p = plt.gcf()
            p.gca().add_artist(my_circle)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting donut: {e}")
            tastyPlots.show_fail_gif()

    @staticmethod
    @validate_dataframe
    def chocolateBar(df, title="Chocolate Bars Plot", ylabel="Chocolate Bars y label"):
        """
        Generate and display a chocolate bar chart.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the chocolate bar chart.
        - title (str): Title of the chocolate bar chart.
        - ylabel (str): Y-axis label of the chocolate bar chart.
        """
        try:
            # Plot Bars
            plt.figure(figsize=(16, 10), dpi=80)
            plt.bar(df.iloc[:, 0], df.iloc[:, 1], color="#65350F", width=.5)
            for i, val in enumerate(df['counts'].values):
                plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom',
                         fontdict={'fontweight': 500, 'size': 12})

            # Decoration
            plt.gca().set_xticklabels(df.iloc[:, 0], rotation=60, horizontalalignment='right')
            plt.title(title, fontsize=22)
            plt.ylabel(ylabel)
            plt.ylim(0, 45)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting chocolateBar: {e}")
            tastyPlots.show_fail_gif()

    @staticmethod
    @validate_dataframe
    def someoneTriedToStoleaPieceOfPieButWeStopedThisVillain(df, title="title", legTitle="legTitle"):
        """
        Generate and display a pie chart with an attempted theft theme.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the pie chart.
        - title (str): Title of the pie chart.
        - legTitle (str): Title for the legend of the pie chart.
        """
        try:
            # Draw Plot
            fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=80)

            data = df.iloc[:, 1]
            categories = df.iloc[:, 0]
            explode = [0, 0, 0, 0, 0, 0.1, 0]
            explode = [0] * len(df)
            explode[-1] = 0.1

            def func(pct, allvals):
                absolute = int(pct / 100. * np.sum(allvals))
                return "{:.1f}% ({:d} )".format(pct, absolute)

            wedges, texts, autotexts = ax.pie(data,
                                              autopct=lambda pct: func(pct, data),
                                              textprops=dict(color="w"),
                                              colors=plt.cm.Dark2.colors,
                                              startangle=140,
                                              explode=explode)

            # Decoration
            ax.legend(wedges, categories, title=legTitle, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.setp(autotexts, size=10, weight=700)
            ax.set_title(title)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting someoneTriedToStoleaPieceOfPieButWeStopedThisVillain: {e}")
            tastyPlots.show_fail_gif()

    @staticmethod
    @validate_dataframe
    def someoneActualyStoledPieceOfPie(df, title="title", legTitle="legTitle"):
        """
        Generate and display a pie chart with a successful theft theme.

        Parameters:
        - df (pd.DataFrame or pd.Series): Input data for the pie chart.
        - title (str): Title of the pie chart.
        - legTitle (str): Title for the legend of the pie chart.
        """
        try:
            # Draw Plot
            fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=80)

            data = df.iloc[:, 1]
            categories = df.iloc[:, 0]
            explode = [0, 0, 0, 0, 0, 0.1, 0]
            explode = [0] * len(df)
            explode[-1] = 10

            def func(pct, allvals):
                absolute = int(pct / 100. * np.sum(allvals))
                return "{:.1f}% ({:d} )".format(pct, absolute)

            wedges, texts, autotexts = ax.pie(data,
                                              autopct=lambda pct: func(pct, data),
                                              textprops=dict(color="w"),
                                              colors=plt.cm.Dark2.colors,
                                              startangle=140,
                                              explode=explode)

            # Decoration
            ax.legend(wedges, categories, title=legTitle, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.setp(autotexts, size=10, weight=700)
            ax.set_title(title)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting someoneActualyStoledPieceOfPie: {e}")
            tastyPlots.show_fail_gif()
