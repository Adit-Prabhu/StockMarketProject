import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.style.use("fivethirtyeight")


def plot_closing(plot_type, company_list, tech_list):
    # Let's see a historical view of the plot_type
    plt.figure(figsize=(12, 6))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    # Plot the Adj Close prices for the given tech stocks
    if plot_type == "Closing":
        for i, company in enumerate(company_list, 1):
            plt.subplot(2, 2, i)
            company['Adj Close'].plot()
            plt.title(f"Closing Price of {tech_list[i - 1]}")
            plt.ylabel("Adj Close")
            plt.xlabel(None)
    # Plot the Volume for the given tech stocks
    elif plot_type == "Volume":
        for i, company in enumerate(company_list, 1):
            plt.subplot(2, 2, i)
            company['Volume'].plot()
            plt.title(f"Volume of {tech_list[i - 1]}")
            plt.ylabel("Volume")
            plt.xlabel(None)

    # Show the plot
    plt.tight_layout()
    plt.show()
