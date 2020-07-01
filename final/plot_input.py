import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from csv import reader
import seaborn as sns
import json
print("----------------------------------")

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']

names = []  # 名字
All_inputs = []  # 存放輸入
All_output = []  # 存放輸出


# 取得一些顏色
color = []

for i in range(5):
    line, = plt.plot(0, 0)
    color.append(line.get_color())

plt.clf()
plt.close()

# 載入 CSV 檔案
with open(r'celebrity.csv', 'r') as f:  # 檔案需放在同個目錄下
    data = list(reader(f))
    for i in data[1:]:
        names.append(i[0].split('\n')[0])
        All_inputs.append(i[1:-1])
        All_output.append(i[-1])

# 看一下輸入輸出資料有沒有進去

All_inputs = np.array(All_inputs).astype(float).astype(int)
All_output = np.array(All_output).astype(float).astype(int)
print(All_inputs)
print(All_output)


taiwan = All_inputs[All_output == 1]
world = All_inputs[All_output == 2]


def year_plot():
    """ 繪製出生年與出道年的圖表 """
    plt.scatter(taiwan[:, 0], taiwan[:, 3] -
                taiwan[:, 0], c=color[0], label='台灣')
    plt.scatter(world[:, 0], world[:, 3]-world[:, 0], c=color[1], label='世界')
    plt.grid()
    plt.xlabel('出生年')
    plt.ylabel('出道年-出生年')
    plt.legend()
    plt.show()


def gender_plot():
    """ 繪製性別長條圖 """
    N = 2
    labels = ['台灣', '世界']

    men = [np.count_nonzero(2-taiwan[:, 1]), np.count_nonzero(2-world[:, 1])]
    women = [np.count_nonzero(taiwan[:, 1]-1), np.count_nonzero(world[:, 1]-1)]

    print(men)
    print(women)

    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    plt.ylabel('人數')
    plt.xticks(ind, labels)

    rects1 = plt.bar(ind - width/2, men, width, label='男')
    rects2 = plt.bar(ind + width/2, women, width, label='女')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.legend()

    for rects in [rects1, rects2]:
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.tight_layout()

    plt.show()

def genter_pie():
    """ 繪製性別圓餅圖 """
    men = np.array([np.count_nonzero(2-taiwan[:, 1]), np.count_nonzero(2-world[:, 1])])/20
    women = np.array([np.count_nonzero(taiwan[:, 1]-1), np.count_nonzero(world[:, 1]-1)])/20
    plt.subplot(121)
    plt.pie([men[0], women[0]], labels=['男生', '女生'], autopct='%1.1f%%')
    plt.title('台灣')
    plt.subplot(122)
    plt.pie([men[1], women[1]], labels=['男生', '女生'], autopct='%1.1f%%')
    plt.title('世界')
    plt.show()



def month_plot():
    """ 繪製出到月份分布圖 """
    N = 12
    labels = ['一月', '二月', '三月', '四月', '五月', '六月',
              '七月', '八月', '九月', '十月', '十一月', '十二月']

    ctaiwan = []
    cworld = []
    call = []

    for i in range(1, 12+1):
        ctaiwan.append(np.count_nonzero(taiwan[:, 4] == i))
        cworld.append(np.count_nonzero(world[:, 4] == i))
        call.append(np.count_nonzero(All_inputs[:, 4] == i))

    print(ctaiwan)
    print(cworld)

    ind = np.arange(N)    # the x locations for the groups
    width = 0.3       # the width of the bars: can also be len(x) sequence

    plt.ylabel('人數')
    plt.xticks(ind, labels)

    rects1 = plt.bar(ind - width, ctaiwan, width, label='台灣')
    rects2 = plt.bar(ind, cworld, width, label='世界')
    rects3 = plt.bar(ind + width, call, width, label='加總')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.legend()

    for rects in [rects1, rects2, rects3]:
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.annotate('{}'.format(height),
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.tight_layout()

    plt.show()


def distribution_plot(id, label, log=False, bins=4):
    """ 繪製分布圖 id 為取用的資料欄位，label 為圖表名稱，log 為是否用對數刻度 """
    fig, ax = plt.subplots()

    sns.distplot(taiwan[:, id], label='台灣', bins=bins, color=color[0])
    if log:
        plt.xscale('log')
    plt.xlabel(label)

    plt.yticks([])

    ax = plt.twinx()
    sns.distplot(world[:, id], label='世界', bins=bins, color=color[1])
    fig.legend(loc="upper right", bbox_to_anchor=(
        1, 1), bbox_transform=ax.transAxes)

    plt.yticks([])

    print(min(taiwan[:, id]))
    print(min(world[:, id]))
    print(max(taiwan[:, id]))
    print(max(world[:, id]))
    print(np.average(taiwan[:, id]))
    print(np.average(world[:, id]))

    plt.show()


def saveToJSON():
    """ 將表格存為 JSON 格式 """
    with open('./data.json', 'w') as f:
        data = []
        for i, name in enumerate(names):
            data.append({
                "id": i,
                "name": name,
                "birthYear": All_inputs[i][0],
                "gender": All_inputs[i][1],
                "fans": All_inputs[i][2],
                "debutYear":  All_inputs[i][3],
                "debutMonth":  All_inputs[i][4],
                "isTaiwan":  All_output[i]
            })
        json.dump(data, f)

""" 下面把想要執行的註解拿掉就可以了 """
# saveToJSON()
# year_plot()
# gender_plot()
# distribution_plot(2, "粉絲數", True)
# distribution_plot(0, "出生年")
# month_plot()
# genter_pie()
