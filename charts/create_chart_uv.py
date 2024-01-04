import matplotlib.pyplot as plt
import numpy as np
import os
import json

cat_list = ["chips", "juice", "paste", "pringles", "shampoo", "teabox", "pastry"]
rgb_error_masks = np.zeros((7,80000*7, 4))
rgb_error_alls = np.zeros((7,80000*7, 4))
bop_dir = "/dataset/deform_dataset"
i = 0
for scene_number in os.listdir(os.path.join(bop_dir, "rgb_error")):
    with open(os.path.join(bop_dir,'rgb_error',scene_number), "r") as json_file:
        results = json.load(json_file)
    
    for index, r in enumerate(results):
        obj = r["obj"]
        rgb_error_mask = r["rgb_error_mask"]
        rgb_error_all = r["rgb_error_all"]

        rgb_error_mask.insert(0, np.mean(rgb_error_mask))
        rgb_error_all.insert(0, np.mean(rgb_error_all))

        rgb_error_masks[cat_list.index(obj)][i] = np.array(rgb_error_mask)
        rgb_error_alls[cat_list.index(obj)][i] = np.array(rgb_error_all)
        i += 1

for index, cat in enumerate(cat_list):
    # rgb chips box plot
    x = np.linspace(1, 4, 4)
    fig, ax = plt.subplots(figsize=(14,9))
    plt.yticks(fontsize=16)
    plt.xticks(x, fontsize=16)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'RGB' 
    labels[1] = 'R' 
    labels[2] = 'G' 
    labels[3] = 'B'
    ax.set_xticklabels(labels)
    # rot box plot
    rs = []
    rgbs = []
    bs = []
    gs = []
    for line in rgb_error_masks[index]:
        rgb, r, g, b = line
        if rgb > 0:
            rgbs.append(rgb)
            rs.append(r)
            gs.append(g)
            bs.append(b)
    data = [rgbs, rs, gs, bs]
    bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                    boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=labels,
                    medianprops = dict(color = "blue", linewidth = 2),
                    meanprops = dict(color = "green", linewidth = 2),
                    capprops = dict(color = "black", linewidth = 2),
                    whiskerprops = dict(color = "black", linewidth = 2))
    plt.xlabel("RGB channels", fontsize=20)
    plt.ylabel("UV error [-]", fontsize=20)
    plt.title("UV error {}".format(cat), fontsize=30)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
    plt.savefig(os.path.join(bop_dir,"charts","{}_rgb_error.svg".format(cat)))

for index, cat in enumerate(cat_list):
    # rgb chips box plot
    x = np.linspace(1, 4, 4)
    fig, ax = plt.subplots(figsize=(14,9))
    plt.yticks(fontsize=16)
    plt.xticks(x, fontsize=16)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'RGB' 
    labels[1] = 'R' 
    labels[2] = 'G' 
    labels[3] = 'B'
    ax.set_xticklabels(labels)
    # rot box plot
    rs = []
    rgbs = []
    bs = []
    gs = []
    for line in rgb_error_alls[index]:
        rgb, r, g, b = line
        if rgb > 0:
            rgbs.append(rgb)
            rs.append(r)
            gs.append(g)
            bs.append(b)
    data = [rgbs, rs, gs, bs]
    bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                    boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=labels,
                    medianprops = dict(color = "blue", linewidth = 2),
                    meanprops = dict(color = "green", linewidth = 2),
                    capprops = dict(color = "black", linewidth = 2),
                    whiskerprops = dict(color = "black", linewidth = 2))
    plt.xlabel("RGB channels", fontsize=20)
    plt.ylabel("UV error [-]", fontsize=20)
    plt.title("UV error {}".format(cat), fontsize=30)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
    plt.savefig(os.path.join(bop_dir,"charts","{}_rgb_error_all.svg".format(cat)))
