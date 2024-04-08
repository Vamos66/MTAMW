import torch
import random
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import get_data as get_data
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import model
from metrics import multiclass_acc
from tqdm import tqdm
import numpy as np
import pickle
from model import save_model
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def set_random_seed(seed):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed = random.randint(0, 100)
set_random_seed(seed=10)

type_of_data = 'unalign'

if type_of_data == 'align':
    with open(f"../Data/mosei.pkl", "rb") as handle:
        data = pickle.load(handle)

    max_seq_length = 50
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train = get_data.convert_to_features(train_data, tokenizer)
    dev = get_data.convert_to_features(dev_data, tokenizer)
    test = get_data.convert_to_features(test_data, tokenizer)

elif type_of_data == 'unalign':
    try:

        train = load_pickle('data/mosi_pro_train.pkl')
        print("========开始加载训练集=====")
        dev = load_pickle('data/mosi_pro_dev.pkl')
        print("======开始加载验证集=======")
        test = load_pickle('data/mosi_pro_test.pkl')
        print("========开始加载测试集========")

    except:
        print("原始加载数据")
        with open(f"/gemini/data-1/train.pkl", "rb") as handle:
            train_data = pickle.load(handle)
        with open(f"/gemini/data-1/dev.pkl", "rb") as handle:
            dev_data = pickle.load(handle)
        with open(f"/gemini/data-1/test.pkl", "rb") as handle:
            test_data = pickle.load(handle)

        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train = get_data.convert_2_features(train_data, bert_tokenizer, max_tlen=50, max_vlen=70, max_alen=80)
        dev = get_data.convert_2_features(dev_data, bert_tokenizer, max_tlen=50, max_vlen=70, max_alen=80)
        test = get_data.convert_2_features(test_data, bert_tokenizer, max_tlen=50, max_vlen=70, max_alen=80)

        # save
        to_pickle(train, 'data/mosi_pro_train.pkl')
        to_pickle(dev, 'data/mosi_pro_dev.pkl')
        to_pickle(test, 'data/mosi_pro_test.pkl')

elif type_of_data == "ASR":
    with open(f"D:/EDGE/ASR_dataset/asr_speechbrain_train.pkl", "rb") as handle:
        train_data = pickle.load(handle)
    with open(f"D:/EDGE/ASR_dataset/asr_speechbrain_dev.pkl", "rb") as handle:
        dev_data = pickle.load(handle)
    with open(f"D:/EDGE/ASR_dataset/asr_speechbrain_test.pkl", "rb") as handle:
        test_data = pickle.load(handle)
    train = get_data.convert_3_features(train_data, max_tlen=50, max_vlen=70, max_alen=80)
    dev = get_data.convert_3_features(dev_data, max_tlen=50, max_vlen=70, max_alen=80)
    test = get_data.convert_3_features(test_data, max_tlen=50, max_vlen=70, max_alen=80)

print("train的numpy转为tensor")
train_dataset = get_data.get_Dataset(train)
print("dev的numpy转为tensor")
dev_dataset = get_data.get_Dataset(dev)
print("test的numpy转为tensor")
test_dataset = get_data.get_Dataset(test)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True)

dev_dataloader = DataLoader(dev_dataset, batch_size=48, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=48, shuffle=True)

word_size = 50
w_len = 50
v_len = 70
a_len = 80

hidden_size = 768
max_position = 150
modal_size = 3
layer_norm_eps = 1e-05
hidden_dropout_prob = 0.1  # 0.125
word_embedding_dim = 768
visual_dim = 20  # 20 35
audio_dim = 5  # 5  74
attention_dropout_prob = 0.1  # 0.125
num_head = 8
output_attention = 0
num_layer = 3
output_hidden_state = 0
intermediate_size = 768 * 2
num_head_modal = 8
num_layer_modal = 0

config = {'word_size': word_size, 'hidden_size': hidden_size, 'max_position': max_position,
          'word_embedding_dim': word_embedding_dim, 'audio_dim': audio_dim, 'visual_dim': visual_dim,
          'modal_size': modal_size, 'layer_norm_eps': layer_norm_eps, 'hidden_dropout_prob': hidden_dropout_prob,
          'attention_dropout_prob': attention_dropout_prob, 'num_head': num_head, 'output_attention': output_attention,
          'num_layer': num_layer, 'output_hidden_state': output_hidden_state, 'intermediate_size': intermediate_size,
          'num_head_modal': num_head_modal, 'num_layer_modal': num_layer_modal, 'w_len': w_len, 'v_len': v_len,
          'a_len': a_len}

TF_model = model.My_model(config).to(DEVICE)
"""
TF_model = model.My_model(config)
#冻结参数或者加入噪声
noise_lambda = 0.2
freeze_layers = ['bert.embeddings', 'bert.encoder', 'bert.pooler']
for name, param in TF_model.named_parameters():
    # param.requires_grad = False
    for ele in freeze_layers:
        if ele in name:
            TF_model.state_dict()[name][:] += (torch.rand(param.size())-0.5) * noise_lambda * torch.std(param)
            #param.requires_grad = False
            break
TF_model.to(DEVICE)
"""

num_train_optimization_steps = (
        int(
            len(train_dataset) / 48 / 1  #####dev---->train
        )
        * 40
)

weight_params = []
other_params = []
weight_lr = 0.001
other_lr = 6e-5

for name, para in TF_model.named_parameters():
    if para.requires_grad:
        if "w_" in name:
            weight_params += [para]
        else:
            other_params += [para]
params = [
    {"params": weight_params, "lr": weight_lr},
    {"params": other_params, "lr": other_lr},
]
gradient_accumulation_step = 1
# optimizer = AdamW(params= TF_model.parameters(), lr=6e-5,weight_decay=0) #mosi = 6e-5
optimizer = AdamW(params, weight_decay=0)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * num_train_optimization_steps,
    num_training_steps=num_train_optimization_steps,
)


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, input_mask, my_mask, segment_ids, visual_embedding, audio_embedding, label_ids = batch
        input_ids = torch.squeeze(input_ids, 1)
        input_mask = torch.squeeze(input_mask, 1)
        my_mask = torch.squeeze(my_mask, 1)
        segment_ids = torch.squeeze(segment_ids, 1)
        visual_embedding = torch.squeeze(visual_embedding, 1)
        audio_embedding = torch.squeeze(audio_embedding, 1)
        outputs = model(
            input_ids,
            input_mask,
            my_mask,
            segment_ids,
            visual_embedding,
            audio_embedding
        )
        logits = outputs
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if gradient_accumulation_step > 1:
            loss = loss / gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, input_mask, my_mask, segment_ids, visual_embedding, audio_embedding, label_ids = batch
            input_ids = torch.squeeze(input_ids, 1)
            input_mask = torch.squeeze(input_mask, 1)
            my_mask = torch.squeeze(my_mask, 1)
            segment_ids = torch.squeeze(segment_ids, 1)
            visual_embedding = torch.squeeze(visual_embedding, 1)
            audio_embedding = torch.squeeze(audio_embedding, 1)
            outputs = model(
                input_ids,
                input_mask,
                my_mask,
                segment_ids,
                visual_embedding,
                audio_embedding
            )

            logits = outputs

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if gradient_accumulation_step > 1:
                loss = loss / gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, input_mask, my_mask, segment_ids, visual_embedding, audio_embedding, label_ids = batch
            input_ids = torch.squeeze(input_ids, 1)
            # tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))
            input_mask = torch.squeeze(input_mask, 1)
            my_mask = torch.squeeze(my_mask, 1)
            segment_ids = torch.squeeze(segment_ids, 1)
            visual_embedding = torch.squeeze(visual_embedding, 1)
            audio_embedding = torch.squeeze(audio_embedding, 1)
            outputs = model(
                input_ids,
                input_mask,
                my_mask,
                segment_ids,
                visual_embedding,
                audio_embedding
            )

            logits = outputs

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):
    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
    return acc, mae, corr, f_score, mult_a7


max_acc = []
max_acc1 = []
val_loss = 0
######## 控制是否需要训练
Train = 1
if Train == True:
    for epoch_i in range(40):

        print("EPOCH", epoch_i + 1)
        a = train_epoch(TF_model, train_dataloader, optimizer, scheduler)
        b = eval_epoch(TF_model, dev_dataloader)
        print("TRAIN-LOSS:", a)
        print("EVAL-LOSS:", b)
        # 测试集False_or_false
        acc, mae, corr, f_score, mult_a7 = test_score_model(TF_model, test_dataloader, False)
        print("ACC:", acc)
        print("MAE:", mae)
        print("CORR:", corr)
        print("F_SCORE:", f_score)
        print("ACC7:", mult_a7)
        print(TF_model.state_dict()['transformer.Layer.0.attention.w_a'],
              TF_model.state_dict()['transformer.Layer.1.attention.w_a'],
              TF_model.state_dict()['transformer.Layer.2.attention.w_a'])
        print(TF_model.state_dict()['transformer.Layer.0.attention.w_b'],
              TF_model.state_dict()['transformer.Layer.1.attention.w_b'],
              TF_model.state_dict()['transformer.Layer.2.attention.w_b'])
        print(TF_model.state_dict()['transformer.Layer.0.attention.w_c'],
              TF_model.state_dict()['transformer.Layer.1.attention.w_c'],
              TF_model.state_dict()['transformer.Layer.2.attention.w_c'])
        if acc >= val_loss:
            val_loss = acc
        save_model(TF_model, acc, epoch_i + 1)
        max_acc.append(acc)
        if epoch_i + 1 == 40:
            pass
            # save_model(TF_model,acc,epoch_i+1)
    print(max(max_acc))

if Train == 0:
    TF_model = model.My_model(config).to(DEVICE)
    model = TF_model
    model.load_state_dict(torch.load('./weights/mosei_model.pt')['state_dict'])
    acc, mae, corr, f_score, mult_a7 = test_score_model(TF_model, test_dataloader, False)
    acc_1, mae_1, corr_1, f_score_1, mult_a7_1 = test_score_model(TF_model, test_dataloader, True)
    print("ACC:", acc)
    print("MAE:", mae)
    print("CORR:", corr)
    print("F_SCORE:", f_score)
    print("ACC7:", mult_a7)
    print(TF_model.state_dict()['transformer.Layer.0.attention.w_a'],
          TF_model.state_dict()['transformer.Layer.1.attention.w_a'],
          TF_model.state_dict()['transformer.Layer.2.attention.w_a'])
    print(TF_model.state_dict()['transformer.Layer.0.attention.w_b'],
          TF_model.state_dict()['transformer.Layer.1.attention.w_b'],
          TF_model.state_dict()['transformer.Layer.2.attention.w_b'])
    print(TF_model.state_dict()['transformer.Layer.0.attention.w_c'],
          TF_model.state_dict()['transformer.Layer.1.attention.w_c'],
          TF_model.state_dict()['transformer.Layer.2.attention.w_c'])
    print("ACC:", acc_1)
    print("MAE:", mae_1)
    print("CORR:", corr_1)
    print("F_SCORE:", f_score_1)
    print("ACC7:", mult_a7_1)




