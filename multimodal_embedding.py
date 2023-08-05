import torch
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer,BertPreTrainedModel
import numpy as np
from transformer4onemodal import Mul_Encoder



class MultimodalEmbedding(nn.Module):
    """
    构建word_embedding, position_embedding , modal_embedding
    """
    def __init__(self, config):
        super(MultimodalEmbedding , self).__init__()
        self.modal_size = config['modal_size']        #模态数
        self.v_len = config['v_len']
        self.a_len = config['a_len']
        self.w_len = config['w_len']
        self.position_embedding = nn.Embedding(int(config['max_position']/config['modal_size']) , config['hidden_size'])
        self.posi_visual_embedding = nn.Embedding(config['v_len'], config['hidden_size'])
        self.posi_audio_embedding = nn.Embedding(config['a_len'], config['hidden_size'])
        self.modal_embedding = nn.Embedding(config['modal_size'] , config['hidden_size'])

        self.bert = BertModel.from_pretrained('bert-base-uncased' , output_hidden_states = True )

        self.esp_1 = nn.Parameter(torch.zeros(1, 1, config['visual_dim']))  #20 35
        self.esp_2 = nn.Parameter(torch.zeros(1, 1 , config['audio_dim'])) #5  74
        self.fc_visual = nn.Linear(config['visual_dim'] , config['word_embedding_dim'])
        #self.dropout_v = nn.Dropout(p=0.1)
        self.fc_audio = nn.Linear(config['audio_dim'], config['word_embedding_dim'])
        #self.dropout_a = nn.Dropout(p=0.1)

        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])


        self.encoder_v = Mul_Encoder(config)
        self.encoder_a = Mul_Encoder(config)
        ###################################


    def forward(self , input_ids , input_mask, segment_ids, visual_embedding , audio_embedding):
        size_of_batch = input_ids.size(0)

        #cls_token = self.cls_token.repeat(size_of_batch, 1, 1)
        esp_1 = self.esp_1.repeat(size_of_batch, 1, 1)
        esp_2 = self.esp_2.repeat(size_of_batch, 1, 1)

        visual_embedding = torch.cat([esp_1, visual_embedding], dim=1)
        audio_embedding = torch.cat([esp_2, audio_embedding], dim=1)


        #word_embedding = torch.cat([cls_token , word_embedding] , dim= 1)
        word_embedding = self.bert(input_ids = input_ids , attention_mask = input_mask,token_type_ids = segment_ids).hidden_states[12]
        visual_embedding = self.fc_visual(visual_embedding)
        audio_embedding = self.fc_audio(audio_embedding)

        #visual position embedding
        visual_position_id = torch.arange(self.v_len, dtype= torch.long , device = visual_embedding.device).unsqueeze(0)
        visual_position_id = visual_position_id.repeat(size_of_batch, 1)
        visual_position_embedding = self.posi_visual_embedding(visual_position_id)

        #audio position embedding
        audio_position_id = torch.arange(self.a_len, dtype= torch.long , device = audio_embedding.device).unsqueeze(0)
        audio_position_id = audio_position_id.repeat(size_of_batch, 1)
        audio_position_embedding = self.posi_audio_embedding(audio_position_id)


        visual_embedding = visual_embedding + visual_position_embedding
        audio_embedding = audio_embedding + audio_position_embedding

        #visual_embedding = self.encoder_v(visual_embedding,input_mask)
        #audio_embedding = self.encoder_a(audio_embedding,input_mask)


        input_embedding = torch.cat([word_embedding, visual_embedding, audio_embedding] , dim=1)
        # modal_embedding
        modal_ids_0 = torch.zeros(size=(size_of_batch, self.w_len), dtype= torch.long ,device = input_embedding.device)
        modal_ids_1 = torch.ones(size= (size_of_batch, self.v_len), dtype= torch.long ,device = input_embedding.device)
        modal_ids_2 = 2 * torch.ones(size= (size_of_batch, self.a_len), dtype= torch.long ,device = input_embedding.device)
        modal_ids = torch.cat([modal_ids_0, modal_ids_1 , modal_ids_2] , dim=1)
        modal_embedding = self.modal_embedding(modal_ids)

        embedding = modal_embedding + input_embedding
        embedding = self.LayerNorm(embedding)
        embedding = self.dropout(embedding)

        return embedding









