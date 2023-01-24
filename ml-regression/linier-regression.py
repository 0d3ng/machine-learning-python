import numpy as np
import matplotlib.pyplot as plt


def coef_estimation(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    ss_xy = np.sum(y * x) - n * m_y * m_x
    ss_xx = np.sum(x * x) - n * m_x * m_x
    b_1 = ss_xy / ss_xx
    b_0 = m_y - b_1 * m_x
    return b_0, b_1


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color='m', marker='o', s=30)
    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color='g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def main():
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([100, 300, 350, 500, 750, 800, 850, 900, 1050, 1250])
    b = coef_estimation(x, y)
    print("Estimated coefficient: b_0={} b_1{}".format(b[0], b[1]))
    plot_regression_line(x, y, b)


if __name__ == '__main__':
    main()
