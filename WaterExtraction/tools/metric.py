import numpy as np


# 用于计算混淆矩阵和一系列评估指标，如精度、召回率、F1 分数、交并比等
class Evaluator(object):
    # 初始化评估器，指定类别数量和混淆矩阵的大小
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    # 计算真正例、假正例、真负例、假负例
    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    # 计算精度
    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    # 计算召回率
    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    # 计算 F1 分数
    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    # 计算总体精度
    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    # 计算交并比（IoU）
    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    # Kappa
    def Kappa(self):
        total = np.sum(self.confusion_matrix)
        P_0 = np.sum(np.diag(self.confusion_matrix))
        P_e = np.sum(np.sum(self.confusion_matrix, axis=0) * np.sum(self.confusion_matrix, axis=1)) / total
        kappa = (P_0 - P_e) / (1 - P_e)
        return kappa

    # 根据真实标签和预测标签生成混淆矩阵
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    # 添加一批样本，更新混淆矩阵
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    # 重置混淆矩阵
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)




if __name__ == '__main__':
    # 示例用法，创建一个评估器，添加一批样本，计算指标并打印结果
    gt = np.array([[0, 1, 1],
                   [1, 0, 1],
                   [1, 0, 0]])

    pre = np.array([[0, 1, 1],
                   [1, 0, 1],
                   [0, 1, 1]])

    eval = Evaluator(num_class=2)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.Precision())
    print(eval.Recall())
    print(eval.Intersection_over_Union())
    print(eval.OA())
    print(eval.F1())
    print(eval.Frequency_Weighted_Intersection_over_Union())
    print(eval.Kappa())
