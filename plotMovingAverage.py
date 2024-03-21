import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.style.use("fivethirtyeight")

ma = [10, 20, 50]

def plot_moving_average(company_list, tech_list):
    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        for moving_average in ma:
            company['Adj Close'].rolling(window=moving_average).mean().plot(linewidth=1,label=f"{moving_average} Day MA")
        company['Adj Close'].plot(linewidth=1.5, label="Adj Close")
        plt.title(f"Moving Average of {tech_list[i - 1]}")
        plt.ylabel("Adj Close")
        plt.xlabel(None)
        plt.legend(prop={'size': 6})
    plt.tight_layout()
    plt.show()