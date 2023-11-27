# 对 log.out 文件进行分析，得到每个模型的 perplexity、dev_loss、train_loss、Best score (perplexity)
# 而后做出上述参数与 step 的关系图

import re

import matplotlib.pyplot as plt


def parse_log_file(log_file):
    pattern = r"dev_loss = ([\d.]+)\s+\|\|\s+dev_eval_scores = {'perplexity': ([\d.]+)}"
    perplexity_list = []
    dev_loss_list = []
    train_loss_list = []
    best_score_list = []

    with open(log_file, 'r') as file:
        log_content = file.read()
        matches = re.findall(pattern, log_content)

        for match in matches:
            dev_loss, perplexity = map(float, match)
            perplexity_list.append(perplexity)
            dev_loss_list.append(dev_loss)

        # Extract train_loss and best_score
        train_loss_pattern = r"train_loss = ([\d.]+)"
        best_score_pattern = r"Best score \(perplexity\) = ([\d.-]+)"
        train_loss_matches = re.findall(train_loss_pattern, log_content)
        best_score_matches = re.findall(best_score_pattern, log_content)
        # 删除 best_score_matches 的第一个元素 '-'
        best_score_matches.pop(0)

        for train_loss, best_score in zip(train_loss_matches, best_score_matches):
            train_loss_list.append(float(train_loss))
            best_score_list.append(float(best_score))

    return perplexity_list, dev_loss_list, train_loss_list, best_score_list


# # --------------------------------------------------
# # 分析单一模型
# # --------------------------------------------------
# path = './model/distilgpt2_fine_tuned_coder_3/'
# log_file = path + 'log.out'
#
# perplexity, dev_loss, train_loss, best_score = parse_log_file(log_file)
# # 找到 perplexity 中第一个小于 100 的元素的索引 index
# index = 0
# for i in range(len(perplexity)):
#     if perplexity[i] < 100:
#         index = i
#         break
# # 将 perplexity 中 index 之前的元素的值设置为与 index 相同
# for i in range(index):
#     perplexity[i] = perplexity[index]
# # 将 best_score 中 index 之前的元素的值设置为与 index 相同
# for i in range(index):
#     best_score[i] = best_score[index]
#
# # --------------------------------------------------
# # 分别绘图
# # --------------------------------------------------
# def plot_perplexity(perplexity):
#     steps = list(range(0, len(perplexity) * 200, 200))
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(steps, perplexity)
#
#     plt.xlabel('Training Steps')
#     plt.ylabel('Perplexity')
#     plt.title('Dev Eval Scores vs. Training Steps')
#     plt.grid(True)
#     plt.show()
#
#
# def plot_dev_loss(dev_loss):
#     steps = list(range(0, len(dev_loss) * 200, 200))
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(steps, dev_loss)
#
#     plt.xlabel('Training Steps')
#     plt.ylabel('Dev Loss')
#     plt.title('Dev Loss vs. Training Steps')
#     plt.grid(True)
#     plt.show()
#
#
# def plot_train_loss(train_loss):
#     steps = list(range(0, len(train_loss) * 200, 200))
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(steps, train_loss)
#
#     plt.xlabel('Training Steps')
#     plt.ylabel('Train Loss')
#     plt.title('Train Loss vs. Training Steps')
#     plt.grid(True)
#     plt.show()
#
#
# def plot_best_score(best_score):
#     steps = list(range(0, len(best_score) * 200, 200))
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(steps, best_score)
#
#     plt.xlabel('Training Steps')
#     plt.ylabel('Best Score (Perplexity)')
#     plt.title('Best Score (Perplexity) vs. Training Steps')
#     plt.grid(True)
#     plt.show()
#
#
# plot_perplexity(perplexity)
# plot_dev_loss(dev_loss)
# plot_train_loss(train_loss)
# plot_best_score(best_score)

# --------------------------------------------------
# 分析多个模型
# --------------------------------------------------

path = './model/'
model_list = ['distilgpt2_fine_tuned_coder_1', 'distilgpt2_fine_tuned_coder_2', 'distilgpt2_fine_tuned_coder_3',
              'distilgpt2_fine_tuned_coder_4']
log_file_list = []
for model in model_list:
    log_file_list.append(path + model + '/log.out')

perplexity_list = []
dev_loss_list = []
train_loss_list = []
best_score_list = []
for log_file in log_file_list:
    perplexity, dev_loss, train_loss, best_score = parse_log_file(log_file)
    index = 0
    for i in range(len(perplexity)):
        if perplexity[i] < 100:
            index = i
            break
    for i in range(index):
        perplexity[i] = perplexity[index]
    for i in range(index):
        best_score[i] = best_score[index]
    perplexity_list.append(perplexity)
    dev_loss_list.append(dev_loss)
    train_loss_list.append(train_loss)
    best_score_list.append(best_score)


# --------------------------------------------------
# 绘制在同一张图上
# --------------------------------------------------
def plot(perplexity_list, dev_loss_list, train_loss_list, best_score_list):
    steps = list(range(0, len(perplexity_list[0]) * 200, 200))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, perplexity_list[0], label='perplexity_1')
    plt.plot(steps, perplexity_list[1], label='perplexity_2')
    plt.plot(steps, perplexity_list[2], label='perplexity_3')
    plt.plot(steps, perplexity_list[3], label='perplexity_4')

    plt.xlabel('Training Steps')
    plt.ylabel('Perplexity')
    plt.title('Dev Eval Scores vs. Training Steps')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, dev_loss_list[0], label='dev_loss_1')
    plt.plot(steps, dev_loss_list[1], label='dev_loss_2')
    plt.plot(steps, dev_loss_list[2], label='dev_loss_3')
    plt.plot(steps, dev_loss_list[3], label='dev_loss_4')

    plt.xlabel('Training Steps')
    plt.ylabel('Dev Loss')
    plt.title('Dev Loss vs. Training Steps')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss_list[0], label='train_loss_1')
    plt.plot(steps, train_loss_list[1], label='train_loss_2')
    plt.plot(steps, train_loss_list[2], label='train_loss_3')
    plt.plot(steps, train_loss_list[3], label='train_loss_4')

    plt.xlabel('Training Steps')
    plt.ylabel('Train Loss')
    plt.title('Train Loss vs. Training Steps')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, best_score_list[0], label='best_score_1')
    plt.plot(steps, best_score_list[1], label='best_score_2')
    plt.plot(steps, best_score_list[2], label='best_score_3')
    plt.plot(steps, best_score_list[3], label='best_score_4')

    plt.xlabel('Training Steps')
    plt.ylabel('Best Score (Perplexity)')
    plt.title('Best Score (Perplexity) vs. Training Steps')
    plt.grid(True)
    plt.legend()
    plt.show()


plot(perplexity_list, dev_loss_list, train_loss_list, best_score_list)
