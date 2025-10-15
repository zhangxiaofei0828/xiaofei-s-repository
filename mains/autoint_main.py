import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.AutoInt import AutoInt
import argparse 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import mean_absolute_percentage_error
from utils.utils import set_seed
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from losses.SmoothMapeLoss import SmoothMAPELoss 
from torch.optim.lr_scheduler import CosineAnnealingLR

def label_plot(origin_label_list,log_label_list):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(x=origin_label_list,bins=[0,500,1000,5000,10000,50000],label="origin_label")
    plt.title('the origin label of train data')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(x=log_label_list,bins=[0,2,2.5,3,3.5,4,4.5,5],label="log_label")
    plt.title('the log label of train data')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def progrem(args):
    train_data = pd.read_csv(args.data_path + "train_data_deep.csv",index_col='Unnamed: 0')
    test_data = pd.read_csv(args.data_path + "eval_data_deep.csv",index_col='Unnamed: 0')

    # 数值特征归一化
    if args.num_feature_norm:
        num_feature = [col for col in train_data.columns if col.startswith(args.num_feature_col)] + args.num_derivative_feature
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(train_data[num_feature])
        train_data[num_feature] = scaler.transform(train_data[num_feature])
        test_data[num_feature] = scaler.transform(test_data[num_feature])
    # 目标列 log化
    if args.if_label_log:
        origin_label_list = train_data[args.label].tolist()
        train_data[args.label] = train_data[args.label].apply(lambda x:np.log10(x))
        test_data[args.label] = test_data[args.label].apply(lambda x:np.log10(x))
        log_label_list = train_data[args.label].tolist()
        label_plot(origin_label_list=origin_label_list,log_label_list=log_label_list)
    # 生成模型需要的格式数据
    feature_cols = [col for col in train_data.columns if col != args.label]
    train_df_data = [[j for j in range(len(feature_cols))] for i in range(len(train_data))]
    train_index = pd.DataFrame(data=train_df_data, columns=feature_cols)
    train_index.index = train_data.index

    train_value = train_data[feature_cols]
    train_label = train_data[[args.label]]

    test_df_data = [[j for j in range(len(feature_cols))] for i in range(len(test_data))]
    test_index = pd.DataFrame(data=test_df_data, columns=feature_cols)
    test_index.index = test_data.index

    test_value = test_data[feature_cols]
    test_label = test_data[[args.label]]

    train_index = train_index.astype(np.int32)
    train_value = train_value.astype(np.float32)
    train_label = train_label.astype(np.float32)
    train_index.to_csv(args.data_path + "train_index.csv",index=True)
    train_value.to_csv(args.data_path + "train_value.csv",index=True)
    train_label.to_csv(args.data_path + "train_label.csv",index=True)

    test_index = test_index.astype(np.int32)
    test_value = test_value.astype(np.float32)
    test_label = test_label.astype(np.float32)
    test_index.to_csv(args.data_path + "test_index.csv",index=True)
    test_value.to_csv(args.data_path + "test_value.csv",index=True)
    test_label.to_csv(args.data_path + "test_label.csv",index=True)
    return 


def get_dataloader(args,index,value,label):
    df_len = len(index[0])
    train_len = int(args.train_data_rate * df_len)
    train_index, train_value, train_label = index[0].head(train_len),value[0].head(train_len),label[0].head(train_len)
    valid_index, valid_value, valid_label = index[0].tail(df_len - train_len),value[0].tail(df_len - train_len),label[0].tail(df_len - train_len)

    train_dataset = TensorDataset(torch.from_numpy(train_index.to_numpy()), torch.from_numpy(train_value.to_numpy()), torch.from_numpy(train_label.to_numpy()))
    val_dataset = TensorDataset(torch.from_numpy(valid_index.to_numpy()), torch.from_numpy(valid_value.to_numpy()), torch.from_numpy(valid_label.to_numpy()))
    test_dataset = TensorDataset(torch.from_numpy(index[1].to_numpy()), torch.from_numpy(value[1].to_numpy()), torch.from_numpy(label[1].to_numpy()))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader,val_loader,test_loader

def train_plot(args,train_loss_list,val_loss_list,val_mape_list,lr_list):
    # 画图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='Training Loss')
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率图
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(val_mape_list) + 1), val_mape_list, label='Validation Mape')
    plt.xlabel('Epoch')
    plt.ylabel('Mape')
    plt.title('Validation Mape Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # 绘制动态调整学习率图
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(lr_list) + 1),lr_list, label='Lr')
    plt.xlabel('Epoch')
    plt.ylabel('Lr')
    plt.title('Lr Over Epochs')
    plt.legend()
    plt.grid(True)

    # 显示图形
    plt.tight_layout()
    plt.show()


def train(args,model,train_loader,valid_loader,criterion,optimizer,scheduler):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_loss_list = []
    val_loss_list = []
    val_mape_list = []
    lr_list = []
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_data_loss = 0
        train_l2_loss = 0
        train_size = 0
        for inputs1, inputs2, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2)  # 传入两个输入
            loss = criterion(outputs, targets)
            train_data_loss += loss.item() * inputs1.size(0)
            train_size += inputs1.size(0)
            # deep l2正则化
            if args.deep_layers != None and args.l2_lambda > 0:
                l2_reg = torch.tensor(0.0,requires_grad=True)
                for name,param in model.named_parameters():
                    if 'deep' in name:
                        l2_reg = l2_reg + torch.norm(param,p=2)
                loss += args.l2_lambda * l2_reg
                train_l2_loss += args.l2_lambda * l2_reg.detach().numpy().item()
            loss.backward()
            optimizer.step()

        train_data_loss /= train_size
        train_l2_loss /= len(train_loader)
        train_loss_list.append(train_data_loss + train_l2_loss)

        # 验证阶段
        model.eval()
        val_data_loss = 0
        val_size = 0
        val_l2_loss = 0
        val_pre = []
        val_real = []
        with torch.no_grad():
            for inputs1, inputs2, targets in valid_loader:
                outputs = model(inputs1, inputs2)  # 传入两个输入
                loss = criterion(outputs, targets)
                val_data_loss += loss.item() * inputs1.size(0)
                val_size += inputs1.size(0)
                # deep l2正则化
                if args.deep_layers != None and args.l2_lambda > 0:
                    l2_reg = torch.tensor(0.0,requires_grad=True)
                    for name,param in model.named_parameters():
                        if 'deep' in name:
                            l2_reg = l2_reg + torch.norm(param,p=2)
                    loss += args.l2_lambda * l2_reg
                    val_l2_loss += args.l2_lambda * l2_reg.detach().numpy().item()
                val_pre.extend(outputs.tolist())
                val_real.extend(targets.tolist())

        val_data_loss /= val_size
        val_l2_loss /= len(valid_loader)
        val_mape = mean_absolute_percentage_error(y_true=val_real, y_pred=val_pre)
        val_loss_list.append(val_data_loss + val_l2_loss)
        val_mape_list.append(val_mape)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)
        # 打印训练和验证损失
        print(f'Epoch {epoch+1}/{args.epochs}, Current lr:{current_lr:.4f}, Train data Loss: {train_data_loss:.4f}, Train l2 Loss: {train_l2_loss:.4f}, Val data Loss: {val_data_loss:.4f}, Val l2 Loss: {val_l2_loss:.4f},Val MAPE: {val_mape:.4f}')

        # 检查是否更新最佳验证损失
        if (val_data_loss + val_l2_loss) < best_val_loss:
            best_val_loss = (val_data_loss + val_l2_loss)
            epochs_no_improve = 0
            # 可以在这里保存最佳模型
            torch.save(model.state_dict(), args.model_path + "wzry_model.pth")
        else:
            epochs_no_improve += 1
    
        # 检查是否早停
        if epochs_no_improve >= args.patience:
            print('Early stopping!')
            break
    
    # 画图
    train_plot(args,train_loss_list,val_loss_list,val_mape_list,lr_list)

    return model

def eval(model,test_loader):
    # 验证阶段
    model.eval()
    test_pre = []
    test_real = []
    with torch.no_grad():
        for inputs1, inputs2, targets in test_loader:
            if args.if_label_log:
                outputs = 10 ** model(inputs1, inputs2)
                targets = 10 ** targets
            else:
                outputs = model(inputs1, inputs2)
            outputs = outputs.tolist()
            targets = targets.tolist()
            test_pre.extend(outputs)
            test_real.extend(targets)
    test_mape = mean_absolute_percentage_error(y_true=test_real,y_pred=test_pre)
    # 测试集mape
    print(f'Test MAPE: {test_mape:.4f}')

def predict(model,data):
    pass

def main(args):
    # 固定随机种子
    set_seed(args.seed)

    # 读取并处理数据
    progrem(args)

    # 加工数据
    train_index = pd.read_csv(args.data_path + "train_index.csv",index_col='Unnamed: 0',dtype=np.int32)
    train_value = pd.read_csv(args.data_path + "train_value.csv",index_col='Unnamed: 0',dtype=np.float32)
    train_label = pd.read_csv(args.data_path + "train_label.csv",index_col='Unnamed: 0',dtype=np.float32)

    test_index = pd.read_csv(args.data_path + "test_index.csv",index_col='Unnamed: 0',dtype=np.int32)
    test_value = pd.read_csv(args.data_path + "test_value.csv",index_col='Unnamed: 0',dtype=np.float32)
    test_label = pd.read_csv(args.data_path + "test_label.csv",index_col='Unnamed: 0',dtype=np.float32)

    train_loader,valid_loader,test_loader = get_dataloader(args,[train_index,test_index],[train_value,test_value],[train_label,test_label])

    # 创建模型
    model = AutoInt(args)
    # loss
    # criterion = nn.MSELoss()
    criterion = SmoothMAPELoss(args.smooth_mape_loss_epsilon)
    # optimi
    optimizer = optim.Adam(params=model.parameters(),lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max,eta_min=args.eta_min)  # T_max为周期epoch数

    # 编译模型,优化模型性能，可加速模型训练和推理，但可能不适用动态控制流模型(if-else，for循环等)
    # model = torch.compile(model,backend='inductor')
    # 训练模型
    model = train(args,model,train_loader,valid_loader,criterion,optimizer,scheduler)
    
    # 评估模型
    eval(model,test_loader)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_size",type=int,default=403,help="the num of all feature")
    parser.add_argument("--field_size",type=int,default=403,help="the num of feature field")
    parser.add_argument("--embedding_size",type=int,default=32,help="the size of embedding vector")
    parser.add_argument("--block_shape",type=list,default=[64,64],help="the list of every block's units")
    parser.add_argument("--heads",type=int,default=4,help="the num of multihead-attention's head")
    parser.add_argument("--batch_norm",type=bool,default=False,help="whether to use batch norm")
    parser.add_argument("--batch_norm_momentum",type=float,default=0.1,help="the momentum of batch_norm")
    parser.add_argument("--has_residual",type=bool,default=True,help="whether to use residual in multihead-attention")
    parser.add_argument("--dropout_rate",type=list,default=[0.0,0.0,0.0],help="the list of dropout_rate (multihead-ttention、embedding、deep)")
    parser.add_argument("--deep_layers",type=list,default=None,help="the list of units in deep")
    
    parser.add_argument("--seed",type=int,default=2025,help="the seed of training model")
    parser.add_argument("--epochs",type=int,default=300,help="the epochs of training model")
    parser.add_argument("--t_max",type=int,default=75,help="the epochs of CosineAnnealingLR")
    parser.add_argument("--batch_size",type=int,default=1024,help="the batch_size of training model")
    parser.add_argument("--learning_rate",type=float,default=0.01,help="the learning rate of training model")
    parser.add_argument("--eta_min",type=float,default=0.0005,help="the min learning rate of CosineAnnealingLR")
    parser.add_argument("--l2_lambda",type=float,default=0.0,help="the coefficient of l2 regularization")
    parser.add_argument("--patience",type=int,default=10,help="the patience of early stopping")
    parser.add_argument("--smooth_mape_loss_epsilon",type=float,default=0.001,help="the epsilon of smooth mape loss")

    parser.add_argument("--model_path",type=str,default="data/valuation/",help="the path of model")
    parser.add_argument("--data_path",type=str,default="data/valuation/",help="the path of data")
    parser.add_argument("--train_data_rate",type=float,default=0.9,help="the rate of train data")
    parser.add_argument("--if_label_log",type=bool,default=True,help="whether to log label")
    parser.add_argument("--label",type=str,default="real_price",help="the col of label")
    parser.add_argument("--num_feature_norm",type=bool,default=True,help="whether to use norm num feature")
    parser.add_argument("--num_feature_col",type=str,default="统计信息",help="whether to use norm num feature")
    parser.add_argument("--num_derivative_feature",type=list,default=["缺失特征数","完单距离开服天数","thred_a_price"],help="whether to use norm num feature")
    args = parser.parse_args()

    main(args)


