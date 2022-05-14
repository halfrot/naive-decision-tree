import copy
import sys

import numpy as np


def entropy(x: np.ndarray):
    h = 0
    tot = x.size
    for ele in np.unique(x):
        p = x[x == ele].size / tot
        h -= p * np.log2(p)
    return h


def mutualInformation(x: np.ndarray, y: np.ndarray):
    hy = entropy(y)
    tot = x.size
    for ele in np.unique(x):
        p = x[x == ele].size / tot
        loc = np.where(x == ele)
        hy -= p * entropy(y[loc])
    return hy


class MyDecisionTree:
    def __init__(self, train_path, test_path, max_depth, train_out, test_out, metrics_out):
        self.trainPath = train_path
        self.testPath = test_path
        self.maxDepth = max_depth
        self.trainOut = train_out
        self.testOut = test_out
        self.metricsOut = metrics_out

        self.data = None
        self.tree = None
        self.feature = None
        self.category = None
        self.positiveName = None
        self.negativeName = None

        self.predict_data = None
        self.train_score = 0
        self.test_score = 0

    def build(self):
        init_data = np.array(range(1, len(self.data)))
        self.tree = self.treeSplit(init_data, list(), 0)
        self.printTree(self.tree, self.feature, None, 0)

    def findMost(self, y: np.ndarray):
        maxCate = 0
        maxName = None
        for ele in np.unique(y):
            size = y[y == ele].size
            # print(type(ele))
            if maxCate == size and self.feature[maxCate] < ele:
                maxName = ele
            if maxName is None or maxCate < size:
                maxCate = size
                maxName = ele
        return maxName

    def treeSplit(self, now_data_idx: np.array, ignore_feature_idx: list, depth):
        now_data = self.data[now_data_idx]
        node = dict()
        node["$Type$"] = None
        node["$Child$"] = None
        node["$Data$"] = now_data_idx
        if depth >= self.maxDepth:
            node["$Type$"] = self.category
            y = now_data[:, self.category]
            node["$Child$"] = self.findMost(y)
            return node
        flag = True
        for i in range(1, len(now_data_idx)):
            if now_data[i][self.category] != now_data[i - 1][self.category]:
                flag = False
                break
        if flag:
            node["$Type$"] = self.category
            node["$Child$"] = now_data[0][self.category]
            return node
        flag = True
        for i in range(1, len(now_data_idx)):
            for ele in range(0, len(self.feature)):
                if ele in ignore_feature_idx:
                    continue
                if now_data[i][ele] != now_data[i - 1][ele]:
                    flag = False
                    break
        if flag:
            node["$Type$"] = self.category
            y = now_data[:, self.category]
            node["$Child$"] = self.findMost(y)
            return node
        maxMI = None
        maxFeature = None
        for ele in range(0, len(self.feature)):
            if ele in ignore_feature_idx:
                continue
            nowMI = mutualInformation(now_data[:, ele], now_data[:, self.category])
            if maxMI is None or nowMI > maxMI:
                maxMI = nowMI
                maxFeature = ele
        # if maxFeature == 2:
        #     print("!!!!!!!!!!!!!!!!!")
        now_data_feature = now_data[:, maxFeature]
        node["$Type$"] = maxFeature
        node["$Child$"] = dict()
        new_ignore_feature = ignore_feature_idx + [maxFeature]
        for ele in np.unique(now_data_feature):
            loc = np.where(now_data_feature == ele)
            node["$Child$"][ele] = self.treeSplit(now_data_idx[loc], new_ignore_feature, depth + 1)
        return node

    def findLeaf(self, x: np.ndarray, root):
        if root["$Type$"] == self.category:
            # print("leaf", root["$Child$"])
            return root["$Child$"]
        # print(x, root["$Type$"])
        return self.findLeaf(x, root["$Child$"][x[root["$Type$"]]])

    def predict(self):
        predict = list()
        tot = 0
        correct = 0
        for i in range(1, len(self.predict_data)):
            x = self.predict_data[i][:-1]
            y = self.predict_data[i][self.category]
            predict.append(self.findLeaf(x, self.tree))
            tot += 1
            # print(predict[-1], y)
            if y == predict[-1]:
                correct += 1
        return predict, correct / tot

    def prune(self):
        pass

    def train(self):
        trainFile = open(self.trainPath, "r", encoding="UTF-8")
        self.data = trainFile.readlines()
        trainFile.close()
        for i in range(0, len(self.data)):
            self.data[i] = self.data[i].replace('\n', '').split('\t')
        self.data = np.array(self.data)
        self.feature = self.data[0][:-1]
        self.category = len(self.feature)
        y = self.data[1:, self.category]
        self.positiveName = y[0]
        for ele in y:
            if ele != self.positiveName:
                self.negativeName = ele
                break
        self.build()
        self.predict_data = copy.copy(self.data)
        predict, score = self.predict()
        self.train_score = 1 - score
        outFile = open(self.trainOut, "w", encoding="UTF-8")
        for ele in predict:
            outFile.write(ele + '\n')
        outFile.close()
        # print(predict, 1 - score)

    def test(self):
        testFile = open(self.testPath, "r", encoding="UTF-8")
        self.predict_data = testFile.readlines()
        testFile.close()
        for i in range(0, len(self.predict_data)):
            self.predict_data[i] = self.predict_data[i].replace('\n', '').split('\t')
        self.predict_data = np.array(self.predict_data)
        predict, score = self.predict()
        self.test_score = 1 - score
        outFile = open(self.testOut, "w", encoding="UTF-8")
        for ele in predict:
            outFile.write(ele + '\n')
        outFile.close()
        # print(predict, 1 - score)

    def writeMetrics(self):
        outFile = open(self.metricsOut, "w", encoding="UTF-8")
        outFile.write("error(train): %f\n" % self.train_score)
        outFile.write("error(test): %f\n" % self.test_score)
        outFile.close()

    def printTree(self, root, feature_idx, f_name, depth):
        if depth:
            print("| " * depth, end='')
            print(self.feature[feature_idx] + " = " + f_name + ": ", end='')
        positiveCount = 0
        negativeCount = 0
        y = self.data[root["$Data$"]][:, self.category]
        for ele in y:
            if ele == self.positiveName:
                positiveCount += 1
            else:
                negativeCount += 1
        print("[%d %s/%d %s]" % (positiveCount, self.positiveName, negativeCount, self.negativeName))
        if root["$Type$"] == self.category:
            return
        for ele in root["$Child$"]:
            self.printTree(root["$Child$"][ele], root["$Type$"], ele, depth + 1)
        return


if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    dt = MyDecisionTree(train_path, test_path, max_depth, train_out,
                        test_out,
                        metrics_out)
    dt.train()
    dt.test()
    dt.writeMetrics()
