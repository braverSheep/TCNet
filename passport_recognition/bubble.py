import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from proplot import rc


def generate_bubble_image():

    def circle_area_func(x, p=50, k=150):
        return np.where(x < p, (np.sqrt(x / p) * p) * k, x * k)

    def inverse_circle_area_func(x, p=75, k=150):
        return np.where(x < p * k, (((x / k) / p) ** 2) * p, x / k)

    data = pd.read_excel(r"./experience_data.xlsx", sheet_name="data01")
    model_names = data.Methods
    FLOPs = data.FLOPs
    Params = data.parameters
    values = data.acc
    xtext_positions = [18.1, 3, 18.7, 1, 6, 6, 200, 600, 1810, 360, 975, 10]
    ytext_positions = [60, 82, 77, 80, 75, 85, 83.5, 84, 74, 78, 86.5, 86]

    legend_sizes = [5, 25, 50, 100]
    legend_yposition = 57.5
    legend_xpositions = [3.5, 5.5, 7.8, 10]
    p = 15
    k = 150

    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    
    fig, ax = plt.subplots(figsize=(17, 9), dpi=100, facecolor="w")
    pubble = ax.scatter(x=FLOPs, y=values, s=circle_area_func(Params, p=p, k=k), c=list(range(len(model_names))), cmap=plt.cm.get_cmap("ocean"), lw=3, ec="white", vmin=0, vmax=11)
    center = ax.scatter(x=FLOPs[:-1], y=values[:-1], s=30, c="#e6e6e6")
    ours_ = ax.scatter(x=FLOPs[-1:], y=values[-1:], s=60, marker="*", c="red")

    
    for i in range(len(FLOPs)):
        ax.annotate(model_names[i], xy=(FLOPs[i], values[i]), xytext=(xtext_positions[i], ytext_positions[i]), fontsize=16, fontweight=(200 if i < (len(FLOPs)-1) else 600))
    for i, legend_size in enumerate(legend_sizes):
        ax.text(legend_xpositions[i], legend_yposition, str(legend_size) + "M", fontsize=25, fontweight=200)

    
    kw = dict(prop="sizes", num=legend_sizes, color="#e6e6e6", fmt="{x:.0f}", linewidth=None, markeredgewidth=2, markeredgecolor="green", func=lambda s: np.ceil(inverse_circle_area_func(s, p=p, k=k)))
    legend = ax.legend(*pubble.legend_elements(**kw), bbox_to_anchor=(0.5, 0.15), title="Parameters (Params) / M", ncol=6, fontsize=0, title_fontsize=0, handletextpad=100, frameon=False)

    #ax.set(xlim=(0, 50), ylim=(45, 90), xticks=np.arange(0, 50, step=2), yticks=np.arange(45, 90, step=5), xlabel="GFLOPs", ylabel="Accuracy (acc) / %")
    ax.set(xlim=(0, 23), ylim=(45, 90))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    ax.set_xticks([0,2,4,14,16,18])#,42,44
    ax.set_yticks(np.arange(45, 90, step=5))
    ax.set_xlabel("GFLOPs", fontsize=25)
    ax.set_ylabel("Accuracy (acc) / %", fontsize=25)
    plt.tight_layout()
    plt.savefig("./bubble_image.png", bbox_inches='tight', dpi=300)
    plt.show()



if __name__ == '__main__':

 
    generate_bubble_image()

