import torch
from torch.utils.data import  TensorDataset , DataLoader
from transformers import BertTokenizer
import numpy as np
from transformers import AutoTokenizer, BertModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

max_seq_length = 50
ACOUSTIC_DIM = 74
VISUAL_DIM = 35
TEXT_DIM = 768
MAX_LEN = 50

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, my_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.my_mask = my_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



def prepare_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    #my_mask
    my_mask = np.array(input_mask * 3)
    my_mask = my_mask.reshape(1,-1)
    #my_mask = my_mask.T@ my_mask
    #my_mask = (my_mask == False)

    return input_ids, visual, acoustic, input_mask, my_mask, segment_ids


#需要对齐情况下的数据预处理

def convert_to_features(examples, tokenizer):
    features = []

    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))


        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        # 对单词被拆开成多个时，复制对应的visual 和 audio
        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask,my_mask, segment_ids = prepare_input(
            tokens, visual, acoustic, tokenizer
        )

        # Check input length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert acoustic.shape[0] == max_seq_length - 1
        assert visual.shape[0] == max_seq_length - 1

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                my_mask=my_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,

            )
        )
    return features


#不需要对齐情况下的数据预处理
def convert_2_features(examples, bert_tokenizer, max_tlen, max_vlen, max_alen):
    features = []
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic, actual_words, _vlen, _alen), label_id, idd = example
        text = " ".join(actual_words)
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, max_length=max_tlen, add_special_tokens=True, truncation=True, padding='max_length')
        input_ids = encoded_bert_sent['input_ids']
        input_mask = encoded_bert_sent['attention_mask']
        segment_ids = encoded_bert_sent['token_type_ids']
        len_t = sum(input_mask)

        # Truncate input visual if necessary
        if _vlen >= max_vlen - 1:
            visual = visual[:max_vlen - 1, :]
            len_v = max_vlen
        elif _vlen < max_vlen - 1:
            v_padding = np.zeros((max_vlen - 1 - _vlen, visual.shape[1]))
            visual = np.concatenate((visual, v_padding))
            len_v = _vlen + 1

        # Truncate input acoustic if necessary
        if _alen >= max_alen - 1:
            acoustic = acoustic[:max_alen - 1, :]
            len_a = max_alen
        elif _alen < max_alen - 1:
            a_padding = np.zeros((max_alen - 1 - _alen, acoustic.shape[1]))
            acoustic = np.concatenate((acoustic, a_padding))
            len_a = _alen + 1

        # Check input length
        assert len(input_ids) == max_tlen
        assert len(input_mask) == max_tlen
        assert len(segment_ids) == max_tlen
        assert acoustic.shape[0] == max_alen - 1
        assert visual.shape[0] == max_vlen - 1

        mask_v = np.concatenate((np.ones(len_v), np.zeros(max_vlen - len_v)))
        mask_a = np.concatenate((np.ones(len_a), np.zeros(max_alen - len_a)))


        my_mask = np.concatenate((input_mask, mask_v, mask_a))
        my_mask = my_mask.reshape(1, -1)
        #my_mask = my_mask.T @ my_mask
        #my_mask = (my_mask == False)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                my_mask=my_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features

#处理ASR特征
def convert_3_features(examples, max_tlen, max_vlen, max_alen):
    features = []
    for (ex_index, example) in enumerate(examples):
        #(words, visual, acoustic, actual_words, _vlen, _alen), label_id, idd = example
        (input_ids,input_mask ,segment_ids, visual, acoustic, _vlen, _alen), _label=example
        len_t = sum(input_mask)

        # Truncate input visual if necessary
        if _vlen >= max_vlen - 1:
            visual = visual[:max_vlen - 1, :]
            len_v = max_vlen
        elif _vlen < max_vlen - 1:
            v_padding = np.zeros((max_vlen - 1 - _vlen, visual.shape[1]))
            visual = np.concatenate((visual, v_padding))
            len_v = _vlen + 1

        # Truncate input acoustic if necessary
        if _alen >= max_alen - 1:
            acoustic = acoustic[:max_alen - 1, :]
            len_a = max_alen
        elif _alen < max_alen - 1:
            a_padding = np.zeros((max_alen - 1 - _alen, acoustic.shape[1]))
            acoustic = np.concatenate((acoustic, a_padding))
            len_a = _alen + 1

        # Check input length
        assert len(input_ids) == max_tlen
        assert len(input_mask) == max_tlen
        assert len(segment_ids) == max_tlen
        assert acoustic.shape[0] == max_alen - 1 #return(acoustic.shape[0])
        assert visual.shape[0] == max_vlen - 1

        mask_v = np.concatenate((np.ones(len_v), np.zeros(max_vlen - len_v)))
        mask_a = np.concatenate((np.ones(len_a), np.zeros(max_alen - len_a)))

        my_mask = np.concatenate((input_mask, mask_v, mask_a))
        my_mask = my_mask.reshape(1, -1)
        #my_mask = my_mask.T @ my_mask
        #my_mask = (my_mask == False)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                my_mask=my_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=_label,
            )
        )
    return features


def get_Dataset(data):
    aoli = data
    x0 = 1
    print(x0)
    x0 = x0+1
    all_input_ids = torch.tensor(
        [f.input_ids for f in aoli], dtype=torch.long)
    print(x0)
    x0 = x0+1
    all_input_mask = torch.tensor(
        [f.input_mask for f in aoli], dtype=torch.long)
    print(x0)
    x0 = x0+1
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in aoli], dtype=torch.long)
    print(x0)
    x0 = x0+1
    all_visual = torch.tensor([f.visual for f in aoli], dtype=torch.float)
    print(x0)
    x0 = x0+1
    all_acoustic = torch.tensor(
        [f.acoustic for f in aoli], dtype=torch.float)
    print(x0)
    x0 = x0+1
    all_label_ids = torch.tensor(
        [f.label_id for f in aoli], dtype=torch.float)  
    print(x0)
    x0 = x0+1   
    all_my_mask = torch.tensor(
        [f.my_mask for f in aoli], dtype=torch.float)

    print("=======OK!======")


    dataset = TensorDataset(
        all_input_ids,
        all_input_mask,
        all_my_mask,
        all_segment_ids,
        all_visual,
        all_acoustic,
        all_label_ids,
    )

    return dataset

