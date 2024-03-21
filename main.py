from dataReader import get_data
from plotClosing import plot_closing
from plotMovingAverage import plot_moving_average


def main():
    company_list, tech_list = get_data()
    # plot_closing("Closing", company_list, tech_list)
    # plot_closing("Volume", company_list, tech_list)
    # plot_moving_average(company_list, tech_list)

if __name__ == "__main__":
    main()
