import matplotlib.pyplot as plt
import pandas as pd

class PerformanceMonitor:
    def __init__(self):
        self.performance_data = []

    def log_performance(self, date, portfolio_value):
        self.performance_data.append({"date": date, "portfolio_value": portfolio_value})

    def generate_report(self):
        df = pd.DataFrame(self.performance_data)
        df.set_index("date", inplace=True)
        df.plot()
        plt.title("Portfolio Performance")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.show()