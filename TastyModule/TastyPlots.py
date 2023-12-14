import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pywaffle import Waffle
import random
# Import
class tastyPlots:
    @staticmethod
    def pieChart(df, title="Default Pie Chart"):
        df.plot(kind='pie', subplots=True, figsize=(8, 8))
        plt.title(title)
        plt.ylabel("")
        plt.show()
    @staticmethod
    def lolipop1(df, title="Just lolipops", ylabel = "its a loli y label"):
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

    @staticmethod
    def horiLolipop(df, title="Just lolipops", ylabel = "its a loli y label", xlabel = "its a loli x label"):
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
    @staticmethod
    def wafel(df):
        plt.figure(
            FigureClass=Waffle,
            rows=5,
            columns=20,
            values=df,
            legend={'loc': 'upper left', 'bbox_to_anchor': (1.05, 1)},
        )
        plt.show()
    @staticmethod
    def donut(df):
        # Create a circle at the center of the plot
        my_circle = plt.Circle((0, 0), 0.7, color='white')

        # Custom wedges
        plt.pie(df.iloc[:,1], labels=df.iloc[:,0], wedgeprops={'linewidth': 7, 'edgecolor': 'white'})
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        plt.show()

    @staticmethod
    def chocolateBar(df, title = "Chocolate Bars Plot", ylabel = "Chocolate Bars y label"):
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
    @staticmethod
    def someoneTriedToStoleaPieceOfPieButWeStopedThisVillain(df, title = "title", legTitle = "legTitle"):
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

    @staticmethod
    def someoneActualyStoledPieceOfPie(df, title="title", legTitle="legTitle"):
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



# pieChart example
df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
df = df_raw.groupby('class').size()
tastyPlots.pieChart(df)




#lolipop1 example
df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
df = df_raw[['cty', 'manufacturer']].groupby('manufacturer').apply(lambda x: x.mean())
df.sort_values('cty', inplace=True)
df.reset_index(inplace=True)
print(df)
print(df.index)
print()
print()
print(df.iloc[:, 1])
tastyPlots.lolipop1(df, 'Lollipop Chart for Highway Mileage', 'Miles Per Gallon')




# horizontal loliplot
# Create a dataframe
df = pd.DataFrame({'group': list(map(chr, range(65, 85))), 'values': np.random.uniform(size=20)})
# Reorder it based on the values
ordered_df = df.sort_values(by='values')
print(ordered_df)
tastyPlots.horiLolipop(ordered_df, title="A vertical lolipop plot", xlabel='Value of the variable', ylabel='Group')

# wafelplot
# create simple dummy data
data = {'Kevin': 10, 'Joseph': 7, 'Yan': 8}
# Basic waffle
tastyPlots.wafel(data)


# donut plot
names = ['groupA', 'groupB', 'groupC', 'groupD', 'groupE']
size = [12, 11, 3, 30, 50]
df = pd.DataFrame({'Group': names, 'Size': size})
print(df)

tastyPlots.donut(df)


# chocolate bar chart
# Import Data
df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")
# Prepare Data
df = df_raw.groupby('manufacturer').size().reset_index(name='counts')
print(df)
tastyPlots.chocolateBar(df, title = "Number of Vehicles by Manaufacturers", ylabel = "# Vehicles")


# pie chart when someone has stolen a piece of pie
# Import
df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

# Prepare Data
df = df_raw.groupby('class').size().reset_index(name='counts')
print(df)
tastyPlots.someoneTriedToStoleaPieceOfPieButWeStopedThisVillain(df, title="Class of Vehicles: Pie Chart", legTitle="Vehicle Class")
tastyPlots.someoneActualyStoledPieceOfPie(df, title="Class of Vehicles: Pie Chart", legTitle="Vehicle Class")