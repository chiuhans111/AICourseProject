import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
print("----------------------------------")

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']

names = [] # 名字
All_inputs=[] # 存放輸入
All_output=[] # 存放輸出

# 載入 CSV 檔案
with open(r'celebrity.csv','r') as f: #檔案需放在同個目錄下
    data = list(reader(f))
    for i in data[1:]:
        names.append(i[0].split('\n')[0])
        All_inputs.append(i[1:-1])
        All_output.append(i[-1])


names = np.array(names)
All_inputs = np.array(All_inputs)
All_output = np.array(All_output)




# 看一下輸入輸出資料有沒有進去
print(All_inputs)
print(All_output)



times = 2000
amount = 20

accuracy_log = []

for i in range(times):

    randi = np.arange(40)
    np.random.shuffle(randi)

    # amount = 20

    # 取出一定數量的資料作為訓練用
    X = All_inputs[randi][:amount] #輸入特徵
    Y = All_output[randi][:amount] #標籤
    clf = DecisionTreeClassifier()
    clf.fit(X,Y)




    # 看模型是否有辦法正確預測結果
    pY = clf.predict(All_inputs[randi][amount:])


    correct =  0
    wrong = 0

    for i,name in enumerate(names[randi][amount:]):
        # print(name, "\t為", All_output[randi][amount:][i], "\t預測為", pY[i])
        if All_output[randi][amount:][i] == pY[i]:
            correct += 1
        else:
            wrong += 1
    accuracy = correct/(correct+wrong)
    accuracy_log.append(accuracy)
    # print("正確率", accuracy)
ave_acc = sum(accuracy_log)/times 
plt.subplot(121)
plt.plot(accuracy_log, "o")
plt.ylabel("正確率")
plt.xlabel("實驗次數")
plt.subplot(122)
plt.hist(accuracy_log, bins=10)
plt.xlabel("正確率平均= {:.3f}".format( ave_acc ))
plt.show()

# 這串程式碼會解析樹狀圖
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    # print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "\t" * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}如果 {} 小於等於 {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}否則 {} 大於 {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}  {}".format(indent, ">>台灣成功" if 
                tree_.value[node][0][0]>tree_.value[node][0][1] else ">>世界成功"))

    recurse(0, 0)


tree_to_code(clf,  ['出生年', '性別', '粉絲數', "出道年", '出道月'])
