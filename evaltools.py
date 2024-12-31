import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,hamming_loss
from tabulate import tabulate

def Accuracy(y_true, y_pred):
    p = np.sum(y_true == y_pred)
    return p / y_true.size


def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(y_true.shape[1]): 
         AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP) 

def f1_macro(y_true, y_pred):
    metrics = {
        'Macro-P':[precision_score(y_true, y_pred, average='macro')],
        'Macro-R': [recall_score(y_true, y_pred, average='macro')],
        'Macro-f1':[f1_score(y_true, y_pred, average='macro')]
    }
        
    print_table(metrics, 'Macro')
 
def f1_micro(y_true, y_pred):
    metrics = {
        'Micro-P':[precision_score(y_true, y_pred, average='micro')],
        'Micro-R': [recall_score(y_true, y_pred, average='micro')],
        'Micro-f1':[f1_score(y_true, y_pred, average='micro')]
    }
    print_table(metrics, 'Micro')
 
def f1_weighted(y_true, y_pred):
    metrics = {
        'Weighted-P':[precision_score(y_true, y_pred, average='weighted')],
        'Weighted-R': [recall_score(y_true, y_pred, average='weighted')],
        'Weighted-f1':[f1_score(y_true, y_pred, average='weighted')]
    }
    print_table(metrics, 'Weighted')
 
def print_table(metrics, title):
    print(title)
    print('-' * len(title))
    print(tabulate(metrics, headers='keys', tablefmt='grid'))
 
def overview(y_true, y_pred, y_prob, loss):
    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    map_score = compute_mAP(y_true, y_prob)  
    metrics = {
        'Absolute accuracy rate': [accuracy],
        'Hamming loss': [hamming],
        'mAP': [map_score],
        'accuracy': [Accuracy(y_true, y_pred)],
        'loss': [loss]
    }
    print_table(metrics, 'Overview')

 

def calculate_class_metrics(y_true, y_pred):

    # 获取类别数量
    num_classes = y_true.shape[1]
    
    # 初始化一个字典来存储每个类别的指标
    class_metrics = {
            'Class': list(range(num_classes)),
            'Precision': [],
            'Recall': [],
            'F1 Score': [],
            'Accuracy': []
        }
    
    # 计算并打印每个类别的指标
    for cls in range(num_classes):
        # 将独热编码转换为二进制标签
        true_binary = (y_true[:, cls] == 1).astype(int)
        pred_binary = (y_pred[:, cls] == 1).astype(int)
        
        # 计算精确度、召回率和F1分数
        precision = precision_score(true_binary, pred_binary)
        recall = recall_score(true_binary, pred_binary)  
        f1 = f1_score(true_binary, pred_binary)
        accuracy = accuracy_score(true_binary, pred_binary)
        
        # 存储每个类别的指标
        class_metrics['Precision'].append(precision)
        class_metrics['Recall'].append(recall)
        class_metrics['F1 Score'].append(f1)
        class_metrics['Accuracy'].append(accuracy)
    
    # 打印每个类别的指标
    print("Class Metrics:")
    print("-" * len("Class Metrics:"))
    print(tabulate(class_metrics, headers='keys', tablefmt='grid'))




