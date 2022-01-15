import os
import glob
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torch.nn import functional as F
# from efficientnet_pytorch import EfficientNet
import timm
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.metrics import classification_report
import albumentations as albu
from sklearn.metrics import roc_curve,plot_roc_curve

#ファイルの読み込み
x = np.load('oshusi_xtrain.npy')
y = pd.read_csv('oshusi_ytrain.csv', index_col=0,encoding='SHIFT-JIS')

#次元変換
x= x[:, None, :] #チャネル用に入れる(グレースケールなので1)

#ラベルのみ取り出す
#ラベル名 0:鮪 1:鮭 2:鯛 3:蛸 4:鯵
y = y.iloc[:,1]

# tensor形式へ変換
data = torch.tensor(x, dtype=torch.float32)
target = torch.tensor(y.values, dtype=torch.int64)
y_labels =  np.unique(y.values)

# 目的変数と入力変数をまとめてdatasetに変換
dataset = torch.utils.data.TensorDataset(data,target)

# 各データセットのサンプル数を決定
# train : val : test = 300 : 100 : 100
n_train = 300
n_val = 100
n_test = 100

# データセットの分割
torch.manual_seed(0) #乱数を与えて固定
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val,n_test])

#バッチサイズ
batch_train = 30
batch_val = 10
batch_test = 10

# shuffle はデフォルトで False のため、学習データのみ True に指定
torch.manual_seed(0)

train_loader = torch.utils.data.DataLoader(train, batch_train, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_val)
test_loader = torch.utils.data.DataLoader(test, batch_test)

# 辞書型変数にまとめる(trainとvalをまとめて出す)
dataloaders_dict = {"train": train_loader, "val": val_loader}

class Net(nn.Module):

    # 使用するオブジェクトを定義
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# インスタンス化
net_ori = Net()
net = timm.create_model('efficientnetv2_rw_m',pretrained=True,num_classes=5,in_chans=1)
net2 = timm.create_model('efficientnetv2_rw_m',pretrained=True,num_classes=5,in_chans=1)
net3 = timm.create_model('efficientnetv2_rw_m',pretrained=True,num_classes=5,in_chans=1)

# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 最適化手法の選択
optimizer = torch.optim.Adam(net.parameters())
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

def train_model(net, dataloader_dict, criterion, optimizer, num_epoch,scheduler):

    #入れ子を用意(各lossとaccuracyを入れていく)
    l= []
    a =[]

    # ベストなネットワークの重みを保持する変数
    best_acc = 0.0

    # GPUが使えるのであればGPUを有効化する
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # (エポック)回分のループ
    for epoch in range(num_epoch):
        print('Epoch {}/{} Lr: {:.3e}'.format(epoch + 1, num_epoch,scheduler.get_last_lr()[0]))
        print('-'*20)

        for phase in ['train', 'val']:
            if phase == 'train':
                # 学習モード
                net.train()
            else:
                # 推論モード
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            # 第1回で作成したDataLoaderを使ってデータを読み込む
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 勾配を初期化する
                optimizer.zero_grad()

                # 学習モードの場合のみ勾配の計算を可能にする
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, preds = torch.max(outputs, 1)
                    # 損失関数を使って損失を計算する
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # 誤差を逆伝搬する
                        loss.backward()
                        # パラメータを更新する
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            # 1エポックでの損失を計算
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            # 1エポックでの正解率を計算
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            #lossとaccをデータで保存する
            a_loss = np.array(epoch_loss)
            a_acc = np.array(epoch_acc.cpu()) #GPU⇒CPUに変換してnumpy変換
            a.append(a_acc)
            l.append(a_loss)


            # 一番良い精度の時にモデルデータを保存
            if phase == 'val' and epoch_acc > best_acc:
                print('save model epoch:{:.0f} loss:{:.4f} acc:{:.4f}'.format(epoch,epoch_loss,epoch_acc))
                torch.save(net, 'best_model.pth')

        scheduler.step()  # 追加

    #testとvalのlossとaccを抜き出してデータフレーム化
    a_train = a[::2]
    l_train = l[::2]
    a_train = pd.DataFrame({'train_acc':a_train})
    l_train = pd.DataFrame({'train_loss':l_train})

    a_val = a[1::2]
    l_val = l[1::2]
    a_val = pd.DataFrame({'val_acc':a_val})
    l_val = pd.DataFrame({'val_loss':l_val})

    df_acc = pd.concat((a_train,a_val),axis=1)
    df_loss = pd.concat((l_train,l_val),axis=1)

    #ループ終了後にdfを保存
    df_acc.to_csv('acc.csv', encoding='shift_jis')
    df_loss.to_csv('loss.csv', encoding='shift_jis')

#学習と検証
num_epoch = 4
net = train_model(net, dataloaders_dict, criterion, optimizer, num_epoch,scheduler)

#リスト読み込み
df_a = pd.read_csv('acc.csv', index_col=0,encoding='SHIFT-JIS')
df_l = pd.read_csv('loss.csv', index_col=0,encoding='SHIFT-JIS')

df_a_train = df_a.iloc[:,0].tolist() #train-正解率
df_a_val = df_a.iloc[:,1].tolist() #test-正解率
df_l_train = df_l.iloc[:,0] #train-loss
df_l_val = df_l.iloc[:,1] #test-loss

#結果表示
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1, xlabel='acc', ylabel='epoch') #1つ目のaxを作成
ax2 = fig.add_subplot(1,2,2, xlabel='loss', ylabel='epoch') #2つ目のaxを作成

ax1.plot(df_a_train, 'r-', label='train_acc')
ax1.plot(df_a_val, 'b-', label='val_acc')
ax2.plot(df_l_train, 'r-*', label='train_loss')
ax2.plot(df_l_val, 'b-*', label='train_loss')
ax1.legend()
ax2.legend()
plt.close()

#最適なモデルを呼び出す
best_model = torch.load('best_model.pth')

# 正解率の計算
def test_model(test_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():

        accs = [] # 各バッチごとの結果格納用
        y_preds = []
        y_tests = []

        for batch in test_loader:
            x, t = batch
            x = x.to(device)
            t = t.to(device)
            y = best_model(x)

            y_label = torch.argmax(y, dim=1)
            acc = torch.sum(y_label == t) * 1.0 / len(t)
            accs.append(acc)
            y_preds.append(y_label.to('cpu').detach().numpy().copy())
            y_tests.append(t.to('cpu').detach().numpy().copy())


    # 全体の平均を算出
    avg_acc = torch.tensor(accs).mean()
    std_acc = torch.tensor(accs).std()
    print('Accuracy: {:.1f}%'.format(avg_acc * 100))
    print('Std: {:.4f}'.format(std_acc))

    return y_preds,y_tests
# テストデータで結果確認
y_preds,y_tests = test_model(test_loader)
y_pre = (np.array(y_preds)).flatten()
y_tes = (np.array(y_tests)).flatten()

cm = confusion_matrix(y_tes, y_pre)
cmdf = pd.DataFrame(data=cm, index=[y_labels],
                           columns=[y_labels])
sns.heatmap(cmdf, square=True, cbar=True, annot=True, cmap='Blues')
plt.yticks(rotation=0)
plt.xlabel("Pre", fontsize=13, rotation=0)
plt.ylabel("GT", fontsize=13)
# ax.set_ylim(len(cm), 0)
plt.show()
print(classification_report(y_tes, y_pre))
fpr, tpr, thresholds = roc_curve(val_generator.classes, y_pred_,drop_intermediate=False)

plt.plot(fpr, tpr, marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
print()