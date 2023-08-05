import torch
from torch import nn
import math
import pickle
def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

DEVICE=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Multimodal_SelfAttention(nn.Module):
    def __init__(self , config):
        super(Multimodal_SelfAttention, self).__init__()
        if config['hidden_size'] % config['num_head'] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config['hidden_size'], config['num_head']))

        self.modal_size = config['modal_size']
        self.v_len = config['v_len']
        self.a_len = config['a_len']
        self.w_len = config['w_len']

        self.output_attention = config['output_attention']   #是否要输出注意力的值
        self.num_head = config['num_head']                   #多头注意力机制的数量
        self.attention_head_size = int( config['hidden_size'] / config['num_head'] )  #单头的hidden_size
        self.all_head_size = self.num_head * self.attention_head_size                #多个头concate以后的hidden_size

        self.q = nn.Linear(config['hidden_size'] , self.all_head_size)
        self.k = nn.Linear(config['hidden_size'] , self.all_head_size)
        self.v = nn.Linear(config['hidden_size'] , self.all_head_size)

        self.dropout_1 = nn.Dropout(config['attention_dropout_prob'])

        self.dense = nn.Linear(config['hidden_size'] , config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'] , eps= config['layer_norm_eps'])
        self.dropout_2 = nn.Dropout(config['hidden_dropout_prob'])

        #self.w_a = nn.Parameter(torch.tensor(0.62))
        #self.w_b = nn.Parameter(torch.tensor(0.2))
        #self.w_c = nn.Parameter(torch.tensor(0.2))
        #self.Adap_W = nn.Linear(96,1)

    def transpose1(self, x ):
        new_shape = x.size()[:-1] + (self.num_head , self.attention_head_size)
        x =x.view(*new_shape)        #[batch_size , sentence_len , num_head_size , attention_head_size]
        return x.permute(0,2,1,3)    #[batch_size , num_head_size , sentence_len ,attention_head_size]

    def transpose2(self, x):
        new_shape = x.size()[:-2] +(x.size()[-1] * self.modal_size , int(x.size()[-1] / self.modal_size))
        x = x.view(*new_shape)
        return x

    def transpose3(self , x):
        new_shape = x.size()[:-2] + (x.size()[-1] * self.modal_size , x.size()[-1] * self.modal_size)
        x = x.view(*new_shape)
        return x


    def forward(self, hidden_state , attention_mask ):
        """
        hidden_T = self.transpose1(hidden_state) #[batch_size , num_head_size , sentence_len ,attention_head_size]
        cls_t = hidden_T[:,:,0,:]
        cls_t = torch.unsqueeze(cls_t, dim=2)

        cls_v = hidden_T[:,:,self.w_len,:]
        cls_v = torch.unsqueeze(cls_v,dim = 2)

        cls_a = hidden_T[:,:,self.w_len+self.v_len,:]
        cls_a = torch.unsqueeze(cls_a,dim = 2)

        cls_all = torch.cat((cls_t,cls_v,cls_a),dim=2)

        cls_all = self.Adap_W(cls_all)
        cls_all = torch.squeeze(cls_all)
        weight_all = nn.Softmax(-1)(cls_all)
        wei_t = weight_all[:,:,0]

        #wei_t = torch.unsqueeze(wei_t,dim=2)
        wei_t = wei_t.view(*(wei_t.size()+(1,1)))+0.1
        wei_t = wei_t.repeat(1,1,200,self.w_len)

        wei_v = weight_all[:,:,1]
        wei_v = wei_v.view(*(wei_v.size() + (1, 1)))+0.2
        #wei_v = torch.unsqueeze(wei_v,dim=2)
        wei_v = wei_v.repeat(1,1,200,self.v_len)
        wei_a = weight_all[:,:,2]
        wei_a = wei_a.view(*(wei_a.size() + (1, 1)))+0.2
        #wei_a = torch.unsqueeze(wei_a,dim=2)
        wei_a = wei_a.repeat(1,1,200,self.a_len)
        all_wei = torch.cat((wei_t,wei_v,wei_a),dim=-1)

        :param hidden_state:
        :param attention_mask:
        :return:
        """




        num_head_q = self.q(hidden_state)
        num_head_k = self.k(hidden_state)
        num_head_v = self.v(hidden_state)

        q_layer = self.transpose1(num_head_q)
        k_layer = self.transpose1(num_head_k)
        v_layer = self.transpose1(num_head_v)

        attention_score = torch.matmul(q_layer , k_layer.transpose(-1,-2))
        #print(attention_score.shape)
        attention_score = attention_score / math.sqrt(self.attention_head_size)

        #attention_score = self.transpose2(attention_score)

        #if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.permute(0,2,1)@attention_mask
        my_mask = (attention_mask == False)
        my_mask = my_mask.unsqueeze(1)


        """
        typ = attention_mask.size()[:2] + (attention_mask.size()[-1] * self.modal_size \
                , int(attention_mask.size()[-1]/self.modal_size) )
        attention_mask = attention_mask.view(*typ)        
        """

        attention_score = attention_score.masked_fill(my_mask, -1e9)

        # [batch_size , num_head_size , sentence_len ,sentence_len]

        #删除多模态注意力
        attention_prob = nn.Softmax(-1)(attention_score)
        """
        attention_prob_w = nn.Softmax(-1)(attention_score[:,:,:,:self.w_len])
        attention_prob_v = nn.Softmax(-1)(attention_score[:,:,:,self.w_len:self.w_len+self.v_len])
        attention_prob_a = nn.Softmax(-1)(attention_score[:,:,:,self.w_len+self.v_len:self.w_len+self.v_len+self.a_len])

        attention_prob = torch.cat((attention_prob_w,attention_prob_v,attention_prob_a),dim=-1)        
        """


        #attention_prob = self.transpose3(attention_prob)

        m_a = torch.full((1,1,200,50),1).to(DEVICE)
        m_b = torch.full((1,1,200,70),1).to(DEVICE)
        m_c = torch.full((1,1,200,80),1).to(DEVICE)
        #m_a = self.w_a * torch.ones(1,1,200,50).to(DEVICE)
        #m_b = self.w_b * torch.ones(1,1,200,70).to(DEVICE)
        #m_c = self.w_c * torch.ones(1,1,200,80).to(DEVICE)
        mask2 = torch.cat((m_a,m_b,m_c),dim=3).to(DEVICE)

        attention_prob = attention_prob.mul(mask2)
        #attention_prob = attention_prob.mul(all_wei)

        attention_prob = self.dropout_1(attention_prob)

        context_layer = torch.matmul(attention_prob, v_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        context_layer = self.dense(context_layer)
        context_layer = self.dropout_2(context_layer)
        hidden_state = self.LayerNorm(context_layer + hidden_state)
        #print("-----存储注意力分数-----")
        #print(attention_prob)
        #to_pickle(attention_prob, 'data/attention.pkl')
        outputs = (hidden_state, attention_prob) if self.output_attention else (hidden_state,)
        return outputs



def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



class Feed_Forward(nn.Module):
    def __init__(self, config):
        super(Feed_Forward, self).__init__()

        self.dense_1 = nn.Linear(config['hidden_size'] , config['intermediate_size'])
        self.gelu = nn.GELU()

        self.dense_2 = nn.Linear(config['intermediate_size'] , config['hidden_size'])
        self.Layer_Norm = nn.LayerNorm(config['hidden_size'] , eps= config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])

    def forward(self , hidden_state):

        feed_forward = self.dense_1(hidden_state)
        feed_forward = self.gelu(feed_forward)
        feed_forward = self.dense_2(feed_forward)
        feed_forward = self.dropout(feed_forward)
        feed_forward = self.Layer_Norm(feed_forward + hidden_state)

        return feed_forward


class Multimodal_layer(nn.Module):
    def __init__(self, config):
        super(Multimodal_layer,self).__init__()
        self.attention = Multimodal_SelfAttention(config)
        self.feed_forward = Feed_Forward(config)

    def forward(self , hidden_state , attention_mask):
        attention_output = self.attention(hidden_state , attention_mask)
        output = attention_output[0]
        output = self.feed_forward(output)
        output = (output , ) + attention_output[1:]

        return  output


class Mul_Encoder(nn.Module):
    def __init__(self, config):
        super(Mul_Encoder , self).__init__()
        self.output_attention = config['output_attention']
        self.output_hidden_state = config['output_hidden_state']
        self.Layer = nn.ModuleList([Multimodal_layer(config) for _ in range(config['num_layer'])])

    def forward(self, hidden_state , attention_mask):
        all_hidden_state = ()
        all_attention = ()
        for i, layer_module in enumerate(self.Layer):
            if self.output_hidden_state:
                all_hidden_state = all_hidden_state + (hidden_state,)

            layer_outputs = layer_module(hidden_state, attention_mask)
            hidden_state = layer_outputs[0]

            if self.output_attention:
                all_attention = all_attention + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_state:
            all_hidden_state = all_hidden_state + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_hidden_state:
            outputs = outputs + (all_hidden_state,)
        if self.output_attention:
            outputs = outputs + (all_attention,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)





















