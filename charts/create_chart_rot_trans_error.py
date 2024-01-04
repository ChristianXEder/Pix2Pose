import matplotlib.pyplot as plt
import numpy as np
import os
import json

cat_list = ["chips", "juice", "paste", "pringles", "shampoo", "teabox", "pastry"]
rot_errors = np.zeros((7,80000*7))
trans_errors = np.zeros((7,80000*7))
scores = np.zeros((7,80000*7))
bop_dir = "/dataset/deform_dataset"
i = 0
for scene_number in os.listdir(os.path.join(bop_dir, "result_error")):
    with open(os.path.join(bop_dir,'result_error',scene_number), "r") as json_file:
        results = json.load(json_file)
    
    for index, r in enumerate(results):
        obj = r["obj"]
        score = r["score"]
        rot_error = r["rot_error"]
        trans_error = r["trans_error"]

        rot_errors[cat_list.index(obj)][i] = rot_error
        trans_errors[cat_list.index(obj)][i] = trans_error
        scores[cat_list.index(obj)][i] = score
        i += 1

min_score = np.array([900, 500, 200, 500, 300, 300, 200]) * 2
score_masks = []
for index, cat in enumerate(cat_list):
    score_masks.append(np.where(scores[index] > min_score[index])[0])

score_masks_zero = []
for index, cat in enumerate(cat_list):
    score_masks_zero.append(np.where(scores[index] > 0)[0])

# rot error plot
x = np.linspace(1, 7, 7)
plt.figure(figsize=(14,9))
y = []
e = []
ax = plt.gca()
ax.set_xlim([0.5, 7.5])
labels = [item.get_text() for item in ax.get_xticklabels()]
for index, cat in enumerate(cat_list):
    y.append(np.mean(rot_errors[index][score_masks[index]]))
    e.append(np.std(rot_errors[index][score_masks[index]]))
    labels[index] = cat
plt.errorbar(x, y, yerr=e, fmt='ro', capsize=10, ecolor="green",elinewidth=2, markersize=10)
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Rotation error [°]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Rotation error", fontsize=30)
ax.set_xticklabels(labels)
#ax.set_ylim([0, 0.4])
plt.grid(color = 'lightgrey')
plt.savefig(os.path.join(bop_dir,"charts","rot_error.svg"))

# trans error plot
x = np.linspace(1, 7, 7)
plt.figure(figsize=(14,9))
y = []
e = []
ax = plt.gca()
ax.set_xlim([0.5, 7.5])
labels = [item.get_text() for item in ax.get_xticklabels()]
for index, cat in enumerate(cat_list):
    y.append(np.mean(trans_errors[index][score_masks[index]]))
    e.append(np.std(trans_errors[index][score_masks[index]]))
    labels[index] = cat
plt.errorbar(x, y, yerr=e, fmt='ro', capsize=10, ecolor="green",elinewidth=2, markersize=10)
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Translation error [m]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Translation error", fontsize=30)
ax.set_xticklabels(labels)
#ax.set_ylim([0, 0.4])
#ax.set_xlim([1, 21])
plt.grid(color = 'lightgrey')
plt.savefig(os.path.join(bop_dir,"charts","trans_error.svg"))

# score plot
x = np.linspace(1, 7, 7)
plt.figure(figsize=(14,9))
y = []
e = []
ax = plt.gca()
ax.set_xlim([0.5, 7.5])
labels = [item.get_text() for item in ax.get_xticklabels()]
for index, cat in enumerate(cat_list):
    y.append(np.mean(scores[index][score_masks[index]]))
    e.append(np.std(scores[index][score_masks[index]]))
    labels[index] = cat
plt.errorbar(x, y, yerr=e, fmt='ro', capsize=10, ecolor="green",elinewidth=2, markersize=10)
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Pix2Pose score [-]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Score", fontsize=30)
ax.set_xticklabels(labels)
#ax.set_ylim([0, 0.4])
#ax.set_xlim([1, 21])
plt.grid(color = 'lightgrey')
plt.savefig(os.path.join(bop_dir,"charts","score.svg"))

# rot box plot
data = []
for index, cat in enumerate(cat_list):
    data.append(rot_errors[index][score_masks_zero[index]])
fig, ax = plt.subplots(figsize=(14,9))
ax.set_ylim([0, 185])
bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=cat_list,
                medianprops = dict(color = "blue", linewidth = 2),
                meanprops = dict(color = "green", linewidth = 2),
                capprops = dict(color = "black", linewidth = 2),
                whiskerprops = dict(color = "black", linewidth = 2))
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Rotation error [°]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Rotation error", fontsize=30)
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
plt.savefig(os.path.join(bop_dir,"charts","rot_error_box.svg"))

# trans box plot
data = []
for index, cat in enumerate(cat_list):
    data.append(trans_errors[index][score_masks_zero[index]])
fig, ax = plt.subplots(figsize=(14,9))
ax.set_ylim([0, 0.55])
bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=cat_list,
                medianprops = dict(color = "blue", linewidth = 2),
                meanprops = dict(color = "green", linewidth = 2),
                capprops = dict(color = "black", linewidth = 2),
                whiskerprops = dict(color = "black", linewidth = 2))
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Translation error [m]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Translation error", fontsize=30)
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
plt.savefig(os.path.join(bop_dir,"charts","trans_error_box.svg"))

# score box plot
data = []
for index, cat in enumerate(cat_list):
    data.append(scores[index][score_masks[index]])
fig, ax = plt.subplots(figsize=(14,9))
ax.set_ylim([0, 6100])
bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=cat_list,
                medianprops = dict(color = "blue", linewidth = 2),
                meanprops = dict(color = "green", linewidth = 2),
                capprops = dict(color = "black", linewidth = 2),
                whiskerprops = dict(color = "black", linewidth = 2))
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Pix2Pose score [-]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Score", fontsize=30)
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
plt.savefig(os.path.join(bop_dir,"charts","score_box_filt.svg"))

# rot box plot
data = []
for index, cat in enumerate(cat_list):
    data.append(rot_errors[index][score_masks_zero[index]])
fig, ax = plt.subplots(figsize=(14,9))
ax.set_ylim([0, 185])
bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=cat_list,
                medianprops = dict(color = "blue", linewidth = 2),
                meanprops = dict(color = "green", linewidth = 2),
                capprops = dict(color = "black", linewidth = 2),
                whiskerprops = dict(color = "black", linewidth = 2))
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Rotation error [°]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Rotation error", fontsize=30)
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
plt.savefig(os.path.join(bop_dir,"charts","rot_error_box.svg"))

# trans box plot
data = []
for index, cat in enumerate(cat_list):
    data.append(trans_errors[index][score_masks[index]])
fig, ax = plt.subplots(figsize=(14,9))
ax.set_ylim([0, 0.55])
bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=cat_list,
                medianprops = dict(color = "blue", linewidth = 2),
                meanprops = dict(color = "green", linewidth = 2),
                capprops = dict(color = "black", linewidth = 2),
                whiskerprops = dict(color = "black", linewidth = 2))
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Translation error [m]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Translation error", fontsize=30)
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
plt.savefig(os.path.join(bop_dir,"charts","trans_error_box_filt.svg"))

# score box plot
data = []
for index, cat in enumerate(cat_list):
    data.append(scores[index][score_masks[index]])
fig, ax = plt.subplots(figsize=(14,9))
ax.set_ylim([0, 6100])
bp = ax.boxplot(data, showmeans=True, meanline=True,showfliers=False,patch_artist = True,
                boxprops = dict(facecolor = "lightgrey", linewidth = 2), labels=cat_list,
                medianprops = dict(color = "blue", linewidth = 2),
                meanprops = dict(color = "green", linewidth = 2),
                capprops = dict(color = "black", linewidth = 2),
                whiskerprops = dict(color = "black", linewidth = 2))
plt.xlabel("Scene objects", fontsize=20)
plt.ylabel("Pix2Pose score [-]", fontsize=20)
plt.yticks(fontsize=16)
plt.xticks(x, fontsize=16)
plt.title("Score", fontsize=30)
plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], fontsize="20")
plt.savefig(os.path.join(bop_dir,"charts","score_box_filt.svg"))
