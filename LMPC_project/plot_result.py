# plot_result.py
import matplotlib.pyplot as plt


def plot_result(T, X_log, U_log):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(T, X_log)
    plt.legend(["x1", "x2", "x3", "x4"])
    plt.ylabel("States")

    plt.subplot(2, 1, 2)
    plt.step(T[:-1], U_log)
    plt.ylabel("Input")
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.show()
    plt.savefig("result_plot.png")
    # plt.close()
