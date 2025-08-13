from scipy.stats import ranksums
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def willcoxonfunc(x, y, hue=None, ax=None, fsize=20):
    """Plot the Wilcoxon signed-ran test values (statistic, p-value) in the top left hand corner of a plot."""
    st, p = ranksums(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"$T$ = {st:.2f}", xy=(0.4, 0.6), xycoords=ax.transAxes, fontsize=fsize)
    ax.annotate(f"$p$ = {p:.3f}", xy=(0.38, 0.5), xycoords=ax.transAxes, fontsize=fsize)
    return st, p
def corrfunc(x, y, hue=None, ax=None, fsize=20):
    """Plot the Spearman correlation"""
    st, p = spearmanr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f"$\\rho$ = {st:.2f}", xy=(0.4, 0.6), xycoords=ax.transAxes, fontsize=fsize)
    ax.annotate(f"$p$ = {p:.3f}", xy=(0.38, 0.5), xycoords=ax.transAxes, fontsize=fsize)
    return st, p
def chisqfunc(x, y, hue=None, ax=None, fsize=20):
    """Plot the Wilcoxon signed-ran test values (statistic, p-value) in the top left hand corner of a plot."""
    first_class_x = (x.value_counts()).iloc[0]
    second_class_x = (x.value_counts()).iloc[1]
    first_class_y = (y.value_counts()).iloc[0]
    second_class_y = (y.value_counts()).iloc[1]
    res = chi2_contingency(
        [[first_class_x, second_class_x], [first_class_y, second_class_y]]
    )
    st = res.statistic
    p = res.pvalue

    ax = ax or plt.gca()
    ax.annotate(f"$T$ = {st:.2f}", xy=(0.4, 0.6), xycoords=ax.transAxes, fontsize=fsize)
    ax.annotate(f"$p$ = {p:.3f}", xy=(0.38, 0.5), xycoords=ax.transAxes, fontsize=fsize)
    return st, p