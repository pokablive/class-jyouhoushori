import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser("説明文")
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_epochs",type=int,default=100)
parser.add_argument("--device",type=str,default="cpu")
parser.add_argument("--test_size",type=float,default=0.3)
args = parser.parse_args()

def get_batch(X, Y, batch_size=args.batch_size, shuffle=True):
    """
    バッチごとのデータを生成するジェネレータ関数

    Args:
        X (ndarray): 入力データの特徴量行列
        Y (ndarray): 入力データのラベル行列
        batch_size (int, optional): バッチサイズ. Defaults to args.batch_size.
        shuffle (bool, optional): データをシャッフルするかどうか. Defaults to True.

    Yields:
        tuple: バッチサイズ分の特徴量行列とラベル行列のタプル
    """
    # GPUに対応するために半精度にキャスト
    X = X.astype(np.float32)
    # Xとyをシャッフルする
    index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(index)
        X = X[index]
        Y = Y[index]
    # batch_sizeずつ切り出してタプルとして返すループ
    for i in range(0, Y.shape[0], batch_size):
        x = X[i:i+batch_size]
        y = Y[i:i+batch_size]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        yield (x, y)

def main(lr, batch_size, seed, max_epochs, device, test_size):
    """
    メイン関数

    Args:
        lr (float): 学習率
        batch_size (int): バッチサイズ
        seed (int): 乱数シード値
        max_epochs (int): 最大エポック数
        device (str): デバイス名
        test_size (float): テストデータの割合

    Returns:
        tuple: 訓練データの正解率とテストデータの正解率
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    iris = datasets.load_iris(as_frame=True)["frame"]
    
    #学習データと検証データを分割
    train, test = train_test_split(iris, test_size=test_size)
    
    X_train = train.iloc[:,:4].to_numpy()
    y_train = train.iloc[:,-1].to_numpy()
    X_test = test.iloc[:,:4].to_numpy()
    y_test = test.iloc[:,-1].to_numpy()

    n_labels = len(np.unique(y_train))  # 分類クラスの数 = 3
    y_train = np.eye(n_labels)[y_train] # one hot表現に変換
    y_test = np.eye(n_labels)[y_test]   # one hot表現に変換
    #num_classes = 3
    #y_train = np.eye(num_classes)[y_train.astype(int)][:, :num_classes]
    #y_test = np.eye(num_classes)[y_test.astype(int)][:, :num_classes]
    
   
    net = nn.Sequential(
        nn.Linear(4, 20),
        nn.ReLU(),
        nn.Linear(20, 3),
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    
    # 指定したデバイスへのモデルの転送
    net.to(device)
    
    training_loss = []
    train_acc = []

    for epoch in range(max_epochs):
        net.train() #学習モード
        train_loss = []
        for batch in get_batch(X_train,y_train):
            optimizer.zero_grad() #勾配初期化
            x,y = batch
            x = x.to(device)
            y = y.to(device)
            output = net(x)  #データを流す 
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()   # 重み更新
            train_loss.append(float(loss))
            accuracy = (torch.argmax(output, dim=1) == torch.argmax(y, dim=1)).sum().numpy() / x.size(0) #正解率
            train_acc.append(float(accuracy))
        training_loss.append(np.mean(train_loss))


    test_acc = []
    with torch.no_grad():
        for batch in get_batch(X_test, y_test):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            accuracy = (torch.argmax(output, dim=1) == torch.argmax(y, dim=1)).sum().numpy() / x.size(0)
            test_acc.append(float(accuracy))

    train_Accuracy = np.mean(train_acc)
    test_Accuracy = np.mean(test_acc)

    return train_Accuracy, test_Accuracy

if __name__ == "__main__":
    lr = args.learning_rate
    batch_size = args.batch_size
    seed = args.seed
    max_epochs = args.max_epochs
    device = args.device
    test_size = args.test_size

    train_Accuracy, test_Accuracy = main(lr, batch_size, seed, max_epochs, device, test_size)
    print("train_ACC:", train_Accuracy)
    print("test_ACC:", test_Accuracy)
