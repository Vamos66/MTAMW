import torch
from torch import nn
#from Data import  data
import multimodal_embedding , multimodal_transformer

class My_model(nn.Module):
    def __init__(self,config):
        super(My_model,self).__init__()
        self.embedding = multimodal_embedding.MultimodalEmbedding(config)
        self.transformer = multimodal_transformer.Mul_Encoder(config)
        self.dense = nn.Linear(768, 768)
        self.activation = nn.Tanh()
        self.linear = nn.Linear(768*3,768)
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(768,1)
        self.jihuo = nn.Tanh()
        self.dropout = nn.Dropout(p=0.11)



    def forward(self ,input_ids , input_mask, my_mask,segment_ids, visual_embedding , audio_embedding):

        output = self.embedding(input_ids , input_mask, segment_ids, visual_embedding , audio_embedding)

        output = self.transformer(output,my_mask)[0]
        #output = self.dense(output[:,0])
        output = torch.cat((output[:,0], output[:,50],output[:,120]),dim=1)
        output = self.activation(output)
        output = self.dropout(output)
        output = self.linear(output)
        output = self.jihuo(output)

        output = self.linear2(output)
        output = self.jihuo(output)*3
        return output


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.kaiming_uniform_(m.weight)


def save_model(model,acc,epoch):

    print("Updating the model")
    model_dict = {
        'state_dict': model.state_dict(),
        'acc': acc,
    }
    path = './weights/' + str(epoch)+'model.pt'
    torch.save(model_dict, path)


if __name__ == '__main__':
    word_size = 50
    hidden_size = 768
    max_position = 150
    modal_size = 3
    layer_norm_eps = 0.001
    hidden_dropout_prob = 0.1
    word_embedding_dim = 768
    visual_dim = 47
    audio_dim = 74
    attention_dropout_prob = 0.1
    num_head = 8
    output_attention = 0
    num_layer = 3
    output_hidden_state = 0
    intermediate_size = 768 * 1

    config = {'word_size': word_size, 'hidden_size': hidden_size, 'max_position': max_position,
              'word_embedding_dim': word_embedding_dim, 'audio_dim': audio_dim, 'visual_dim': visual_dim,
              'modal_size': modal_size, 'layer_norm_eps': layer_norm_eps, 'hidden_dropout_prob': hidden_dropout_prob,
              'attention_dropout_prob': attention_dropout_prob, 'num_head': num_head,
              'output_attention': output_attention,
              'num_layer': num_layer, 'output_hidden_state': output_hidden_state,
              'intermediate_size': intermediate_size}
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = My_model(config).to(DEVICE)
    unfreeze_layers = ['bert.embeddings', 'bert.encoder', 'bert.pooler']

    for name, param in model.named_parameters():
        print(name, param.size())

    print("*" * 30)
    print('\n')

    for name, param in model.named_parameters():
        #param.requires_grad = False
        for ele in unfreeze_layers:
            if ele in name:
                param.requires_grad = False
                break
    # 验证一下
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())













