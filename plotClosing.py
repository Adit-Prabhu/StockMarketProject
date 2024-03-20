import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.style.use("fivethirtyeight")


def plot_closing(company_list, tech_list):
    # Let's see a historical view of the closing price
    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company['Adj Close'].plot()
        plt.title(f"Closing Price of {tech_list[i - 1]}")
        plt.ylabel("Adj Close")
        plt.xlabel(None)

    # Show the plot
    plt.tight_layout()
    plt.show()
