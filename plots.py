import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plothastaglinkscounts(dataset):
    mentioncount = []
    linkscount = []
    hashtagcount = []

    categories = ["scientific_claim", "scientific_reference", "scientific_context"]

    for cat in categories:
        subset = dataset[dataset[cat] == 1]
        mentioncount.append(subset["text"].str.count("@").sum())
        linkscount.append(subset["text"].str.count("http").sum())
        hashtagcount.append(subset["text"].str.count("#").sum())

    # --- Plotting the grouped bar chart ---
    x = np.arange(len(categories))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(
        x - width, mentioncount, width, label="Mentions", color="green"
    )
    rects2 = ax.bar(x, linkscount, width, label="Links", color="orange")
    rects3 = ax.bar(
        x + width, hashtagcount, width, label="Hashtags", color="blue"
    )  # Changed color

    # Add some text for labels, title and axes ticks
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height and outlier status."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
            if rects is rects1:
                is_outlier, p_value = check_outlier(mentioncount, height)
                if is_outlier:
                    ax.annotate("Outlier", xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, -15), textcoords="offset points",
                                ha="center", va="top", color='red', fontsize=8)
                    print(f"Category '{categories[i]}' (Mentions): Count={height}, p-value={p_value:.3f} (Outlier)")
                else:
                    print(f"Category '{categories[i]}' (Mentions): Count={height}, p-value={p_value:.3f}")
            elif rects is rects2:
                is_outlier, p_value = check_outlier(linkscount, height)
                if is_outlier:
                    ax.annotate("Outlier", xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, -15), textcoords="offset points",
                                ha="center", va="top", color='red', fontsize=8)
                    print(f"Category '{categories[i]}' (Links): Count={height}, p-value={p_value:.3f} (Outlier)")
                else:
                    print(f"Category '{categories[i]}' (Links): Count={height}, p-value={p_value:.3f}")
            elif rects is rects3:
                is_outlier, p_value = check_outlier(hashtagcount, height)
                if is_outlier:
                    ax.annotate("Outlier", xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, -15), textcoords="offset points",
                                ha="center", va="top", color='red', fontsize=8)
                    print(f"Category '{categories[i]}' (Hashtags): Count={height}, p-value={p_value:.3f} (Outlier)")
                else:
                    print(f"Category '{categories[i]}' (Hashtags): Count={height}, p-value={p_value:.3f}")

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig("rapport/images/hashtag_links_mentions_count_outliers.png", dpi=300)
    plt.show()

def check_outlier(data_lot, single_value, alpha=0.05):
    """
    Checks if a single value is an outlier compared to a data lot using a one-sample t-test.

    Args:
        data_lot (list or numpy.ndarray): The larger dataset to compare against.
        single_value (float or int): The single data point to test.
        alpha (float): The significance level (default is 0.05).

    Returns:
        tuple: (is_outlier (bool), p_value (float))
    """
    if len(data_lot) < 2:
        print("Warning: Data lot has less than 2 data points, outlier detection might be unreliable.")
        return False, np.nan

    mean_lot = np.mean(data_lot)
    std_lot = np.std(data_lot, ddof=1)  # Use sample standard deviation (ddof=1)

    if std_lot == 0:
        print("Warning: Standard deviation of the data lot is zero. Cannot perform t-test.")
        return False, np.nan

    t_statistic = (single_value - mean_lot) / (std_lot / np.sqrt(len(data_lot)))
    degrees_freedom = len(data_lot) - 1
    p_value = stats.t.sf(np.abs(t_statistic), df=degrees_freedom) * 2  # Two-tailed test

    is_outlier = p_value < alpha
    return is_outlier, p_value

if __name__ == "__main__":
    # Load the dataset
    try:
        dataset = pd.read_csv('scitweets_export.tsv', sep='\t')
        plothastaglinkscounts(dataset)
    except FileNotFoundError:
        print("Error: 'scitweets_export.tsv' not found. Please make sure the file is in the correct directory.")
    except Exception as e:
        print(f"An error occurred: {e}")