from dataReader import get_data
from plotClosing import plot_closing


def main():
    company_list, tech_list = get_data()
    plot_closing(company_list, tech_list)

if __name__ == "__main__":
    main()
