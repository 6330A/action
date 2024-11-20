import random

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable

class Matrix(object):
    """ 分类数量，对应的标签 """
    def __init__(self, class_nums, labels):
        self.class_nums = class_nums
        self.labels = labels
        self.matrix = np.zeros((class_nums, class_nums))

    def update(self, pre, label):
        self.matrix[pre, label] += 1

    def summary(self):
        sum_TP = 0
        for i in range(self.class_nums):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("from matrix: the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.class_nums):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        plt.imshow(self.matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.class_nums), self.labels, rotation=45)
        plt.yticks(range(self.class_nums), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = self.matrix.max() / 2
        for i in range(self.class_nums):
            for j in range(self.class_nums):
                # 注意这里的matrix[j, i]不是matrix[i, j]
                info = int(self.matrix[j, i])
                plt.text(i, j, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig("matrix_image.png", dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    m = Matrix(class_nums=4, labels=[chr(i + ord('A')) for i in range(4)])
    for i in range(100):
        x = random.randint(0, m.class_nums - 1)
        y = random.randint(0, m.class_nums - 1)
        m.update(x, y)
    m.plot()
