
import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def plotComparison(self, list1: list(), label1: str,
                    list2: list(), label2: str, 
                    title: str,
                    y_label: str,
                    file_name: str):

        steps = len(list1)
        x = np.linspace(1, steps, num=steps, endpoint=True)
        
        # plot setup
        # plt.style.use("seaborn-dark")
        fig, ax = plt.subplots(dpi=100)
        
        line1, = ax.plot(x, list1)
        line2, = ax.plot(x, list2)
        
        ax.set_xlabel("Epoch", fontweight='bold') 
        ax.set_ylabel(y_label, fontweight='bold') 
        ax.set_title(title, fontweight='bold') 
        ax.legend(handles=[line1, line2], labels=[label1, label2], loc='best')
        plt.savefig(file_name)

   