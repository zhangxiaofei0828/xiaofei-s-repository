import torch
import torch.nn as nn
import torch.nn.functional as F
# 设置默认数据类型为 float32
torch.set_default_dtype(torch.float32)

class MultiHeadAttention(nn.Module):
    def __init__(self,num_fileds = 100,embed_size = 32,num_units=32,num_heads=1,dropout_rate=0,has_residual=True,batch_norm_momentum=0.1):
        super(MultiHeadAttention,self).__init__()
        self.num_fileds = num_fileds
        self.embed_size = embed_size
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.has_residual = has_residual
        self.batch_norm_momentum = batch_norm_momentum

        assert self.num_units % self.num_heads == 0, "num_units needs to be divisible by num_heads"

        # Q: None * self.num_fileds * self.num_units
        self.Q = nn.Linear(in_features=self.embed_size,out_features=self.num_units)
        self.K = nn.Linear(in_features=self.embed_size,out_features=self.num_units)
        self.V = nn.Linear(in_features=self.embed_size,out_features=self.num_units)
        if self.has_residual:
            self.Res_V = nn.Linear(in_features=self.embed_size,out_features=self.num_units)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.normalize = nn.BatchNorm1d(num_features=self.num_fileds,momentum=self.batch_norm_momentum)

    def forward(self, x):
        # queries: None * self.num_fileds * self.embed_size; Q: None * self.num_fileds * self.num_units
        Q = self.relu(self.Q(x))
        K = self.relu(self.K(x))
        V = self.relu(self.V(x))
        if self.has_residual:
            Res_V = self.relu(self.Res_V(x))
        # Q_: (None * self.num_heads) * self.num_fileds * self.num_units // self.num_heads
        Q_ = torch.concat(torch.split(Q,Q.shape[-1] // self.num_heads,dim=2),dim=0)
        K_ = torch.concat(torch.split(K,K.shape[-1] // self.num_heads,dim=2),dim=0)
        V_ = torch.concat(torch.split(V,V.shape[-1] // self.num_heads,dim=2),dim=0)

        # weights: (None * self.num_heads) * self.num_fileds * self.num_fileds
        weights = torch.matmul(Q_,torch.transpose(K_,dim0=1,dim1=2))

        weights = weights / (K_.shape[-1] ** 0.5)
        
        weights = self.softmax(weights)

        # 预防过拟合，论文中没有这个
        weights = self.dropout(weights)

        # outputs: (None * self.num_heads) * self.num_fileds * self.num_units // self.num_heads
        outputs = torch.matmul(weights,V_)
        
        # outputs: None * self.num_fileds * self.num_units
        outputs = torch.concat(torch.split(outputs,outputs.shape[0] // self.num_heads,dim=0),dim=2)
        if self.has_residual:
            outputs += Res_V
        
        output = self.relu(outputs)

        # 预防过拟合，使模型快速收敛，论文中也没有这个
        outputs = self.normalize(output)
        
        return outputs
    

class Deep(nn.Module):
    def __init__(self, deep_input_dim=1024,unit_list=[1024,512,1],batch_norm=True,batch_norm_momentum=0.1,dropout_rate = 0):
        super(Deep,self).__init__()
        self.deep_input_dim = deep_input_dim
        self.unit_list = unit_list
        self.batch_norm = batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        assert self.unit_list[-1] == 1, "the unit of last layer must be 1"

        self.sequence_model = nn.Sequential(nn.Linear(in_features=self.deep_input_dim,out_features=self.unit_list[0],bias=True))
        if self.batch_norm:
            self.sequence_model.append(nn.BatchNorm1d(num_features=self.unit_list[0],momentum=self.batch_norm_momentum))
        self.sequence_model.append(nn.ReLU())
        self.sequence_model.append(nn.Dropout(p=self.dropout_rate))

        for i in range(1,len(self.unit_list) - 1):
            self.sequence_model.append(nn.Linear(in_features=self.unit_list[i-1],out_features=self.unit_list[i],bias=True))
            if self.batch_norm:
                self.sequence_model.append(nn.BatchNorm1d(num_features=self.unit_list[i],momentum=self.batch_norm_momentum))
            self.sequence_model.append(nn.ReLU())
            self.sequence_model.append(nn.Dropout(p=self.dropout_rate))
        
        self.sequence_model.append(nn.Linear(in_features=self.unit_list[-2],out_features=self.unit_list[-1],bias=True))

    def forward(self,x):
        output = self.sequence_model(x)
        return output

class AutoInt(nn.Module):
    def __init__(self, args):
        super(AutoInt,self).__init__()
        self.feature_size = args.feature_size
        self.field_size = args.field_size
        self.embedding_size = args.embedding_size
        
        self.block_shape = args.block_shape
        self.output_size = args.block_shape[-1]
        self.heads = args.heads
        self.batch_norm = args.batch_norm
        self.batch_norm_momentum = args.batch_norm_momentum
        self.has_residual = args.has_residual
        self.dropout_rate = args.dropout_rate

        self.deep_layers = args.deep_layers
        

        self.multihead_attention_sequence = nn.Sequential(
            MultiHeadAttention(
                    num_fileds = self.field_size,
                    embed_size = self.embedding_size,
                    num_units=self.block_shape[0],
                    num_heads=self.heads,
                    dropout_rate=self.dropout_rate[0],
                    has_residual=self.has_residual,
                    batch_norm_momentum=self.batch_norm_momentum
                )
        )

        for i in range(1,len(self.block_shape)):
            self.multihead_attention_sequence.append(
                MultiHeadAttention(
                    num_fileds = self.field_size,
                    embed_size = self.block_shape[i-1],
                    num_units=self.block_shape[i],
                    num_heads=self.heads,
                    dropout_rate=self.dropout_rate[0],
                    has_residual=self.has_residual,
                    batch_norm_momentum=self.batch_norm_momentum
                )
            )
        if self.deep_layers != None:
            self.deep = Deep(
                deep_input_dim=self.field_size * self.embedding_size,
                unit_list=self.deep_layers,
                batch_norm=self.batch_norm,
                batch_norm_momentum=self.batch_norm_momentum,
                dropout_rate = self.dropout_rate[2]
            )

        self.embedding = nn.Embedding(num_embeddings=self.feature_size,embedding_dim=self.embedding_size)
        self.dropout = nn.Dropout(p = self.dropout_rate[1])
        self.predict_layer = nn.Linear(in_features=self.output_size * self.field_size,out_features=1,bias=True)

    def forward(self,x_i,x_v):
        # x_i、x_v: None * self.field_size, x_e: None * self.field_size * self.embedding_size
        x_e = self.embedding(x_i)
        # x_v: None * self.field_size * 1
        x_v = torch.reshape(x_v,shape=[-1,self.field_size,1])
        x_e = torch.multiply(x_e,x_v)
        x_e = self.dropout(x_e)

        if self.deep_layers != None:
            deep_input = torch.reshape(x_e,shape=[-1,self.field_size * self.embedding_size])
            deep = self.deep(deep_input)

        multihead_attention = self.multihead_attention_sequence(x_e)
        predict_input = torch.reshape(multihead_attention,shape=[-1,self.output_size * self.field_size])
        predict = self.predict_layer(predict_input)

        if self.deep_layers:
            predict += deep
        
        return predict
