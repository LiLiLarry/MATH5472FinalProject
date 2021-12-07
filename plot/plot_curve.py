import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.utils import read_res

SAVE_DIR = 'results/figs/'
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_fig3():
    res_softimputeals = read_res(filename='results/res/fig3_ml10m_softimputeALS.txt')
    res_als = read_res(filename='results/res/fig3_ml10m_ALS.txt')

    plt.figure()
    plt.scatter(res_softimputeals[2:, 1], res_softimputeals[2:, 2], label='softImpute-ALS')
    plt.scatter(res_als[2:, 1], res_als[2:, 2], label='ALS')
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR, 'fig3_ml10m.jpg')
    plt.savefig(filename)
    return


def plot_fig1_subfig1():
    res_als = read_res(filename='results/res/fig1_simulation1_ALS.txt')
    res_softimpute = read_res(filename='results/res/fig1_simulation1_softimpute.txt')
    res_softimputeals = read_res(filename='results/res/fig1_simulation1_softimputeALS.txt')

    plt.figure()
    plt.scatter(res_softimputeals[2:, 1], res_softimputeals[2:, 2], c='darkturquoise', label='softImpute-ALS')
    plt.scatter(res_als[2:, 1], res_als[2:, 2], c='orange', label='ALS')
    plt.scatter(res_softimpute[2:, 1], res_softimpute[2:, 2], c='lightgreen', label='softImpute')
    plt.yscale("log")
    plt.legend()
    plt.title('(300, 200), 70% NAs, lambda=120, r=25, rank=15')
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR, 'fig1_simulation1.jpg')
    plt.savefig(filename)
    return


def plot_fig1_subfig2():
    res_als = read_res(filename='results/res/fig1_simulation2_ALS.txt')
    res_softimpute = read_res(filename='results/res/fig1_simulation2_softimpute.txt')
    res_softimputeals = read_res(filename='results/res/fig1_simulation2_softimputeALS.txt')

    plt.figure()
    plt.scatter(res_softimputeals[2:, 1], res_softimputeals[2:, 2], c='darkturquoise', label='softImpute-ALS')
    plt.scatter(res_als[2:, 1], res_als[2:, 2], c='orange', label='ALS')
    plt.scatter(res_softimpute[2:, 1], res_softimpute[2:, 2], c='lightgreen', label='softImpute')
    plt.yscale("log")
    plt.legend()
    plt.title('(800, 600), 90% NAs, lambda=140, r=50, rank=31')
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR, 'fig1_simulation2.jpg')
    plt.savefig(filename)
    return


def plot_fig1_subfig3():
    res_als = read_res(filename='results/res/fig1_simulation3_ALS.txt')
    res_softimpute = read_res(filename='results/res/fig1_simulation3_softimpute.txt')
    res_softimputeals = read_res(filename='results/res/fig1_simulation3_softimputeALS.txt')

    plt.figure()
    plt.scatter(res_softimputeals[2:, 1], res_softimputeals[2:, 2], c='darkturquoise', label='softImpute-ALS')
    plt.scatter(res_als[2:, 1], res_als[2:, 2], c='orange', label='ALS')
    plt.scatter(res_softimpute[2:, 1], res_softimpute[2:, 2], c='lightgreen', label='softImpute')
    plt.yscale("log")
    plt.legend()
    plt.title('(1200, 900), 80% NAs, lambda=300, r=50, rank=27')
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR, 'fig1_simulation3.jpg')
    plt.savefig(filename)
    return


def plot_fig1_subfig4():
    res_als = read_res(filename='results/res/fig1_ml100k_ALS.txt')
    res_softimpute = read_res(filename='results/res/fig1_ml100k_softimpute.txt')
    res_softimputeals = read_res(filename='results/res/fig1_ml100k_softimputeALS.txt')

    plt.figure()
    plt.scatter(res_softimputeals[2:, 1], res_softimputeals[2:, 2], c='darkturquoise', label='softImpute-ALS')
    plt.scatter(res_als[2:, 1], res_als[2:, 2], c='orange', label='ALS')
    plt.scatter(res_softimpute[2:, 1], res_softimpute[2:, 2], c='lightgreen', label='softImpute')
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    filename = os.path.join(SAVE_DIR, 'fig1_ml100k.jpg')
    plt.savefig(filename)
    return


def main():
    plot_fig1_subfig1()
    plot_fig1_subfig2()
    plot_fig1_subfig3()
    plot_fig1_subfig4()
    return


if __name__ == '__main__':
    main()
    print('done!')
