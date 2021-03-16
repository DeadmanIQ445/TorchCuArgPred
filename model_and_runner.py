
# %%
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import json
import numpy as np
from sklearn import preprocessing
from torch import nn
from tqdm.notebook import tqdm
import torch.nn.functional as F


# %%
BATCH_SIZE = 16
load_from_scratch = False
TRAIN_TEST_SPLIT = 0.9
DS_PATH = "/home/deadman445/PycharmProjects/CuArgPred/data/_all_data2.csv"
EPOCHS = 3
shuffle_buffer_size = 10000
SEQ_LENGTH = 512
FREQ_LIMIT = 100
FREQ_CUT_SYMBOL = "<UNK>"
NaN_symbol = ''


# %%

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
bert = AutoModel.from_pretrained("microsoft/codebert-base")
nl_tokens=tokenizer.tokenize("return maximum value")


# %%
import ast

def get_names(src):
    ret = []
    line_lengths = [len(i) for i in src.split('\n')]
    for i in range(1,len(line_lengths)):
        line_lengths[i] += line_lengths[i-1]+1
    line_lengths = [0] + line_lengths
    try:
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.arg):
                ret.append((node.arg,(line_lengths[node.lineno-1]+node.col_offset, line_lengths[node.lineno-1]+node.end_col_offset)))
        return ret
    except:
        print("Could Not process the code")
        return ret

def la(data_batch_i):
  r = []
  for i in data_batch_i:
        if not ((i == NaN_enc[0] or i==FREQ_CUT_ENC[0]) and len(data_batch_i)==1):
            r.append(i)
  if len(r) == 0:
    return pd.NA
  return r


# %%
data = pd.read_csv(DS_PATH)
data['arg_types'] = data['arg_types'].apply(eval)
data = data[data.arg_types.astype(bool)]
df_labels = pd.DataFrame(data['arg_types'].values.tolist())

df_labels[pd.isnull(df_labels)]  = NaN_symbol
df_labels = df_labels.apply(lambda x: x.mask(x.map(x.value_counts())<FREQ_LIMIT, FREQ_CUT_SYMBOL))
enc = preprocessing.LabelEncoder()
all_types = df_labels.apply(pd.Series).stack().values
enc.fit(all_types)
np.save('classes.npy', enc.classes_)
FREQ_CUT_ENC = enc.transform([FREQ_CUT_SYMBOL])
NaN_enc = enc.transform([NaN_symbol])
print(enc.inverse_transform(NaN_enc), enc.inverse_transform(FREQ_CUT_ENC))
print(f'Enc for "NaN" {NaN_enc}, Enc for FREQ_CUT_SYMBOL {FREQ_CUT_ENC}')
df3 = df_labels.apply(enc.transform)
data['labels'] = df3.values.tolist()

data['labels'] = data['labels'].apply(la)
data = data.dropna(subset=['labels'], axis=0)



def train_test_by_repo(data, split=0.75):
    train_l = []
    test_l = []
    c = 0
    train_len = split * len(data)
    for name, i in data.groupby(['repo']).count().sample(frac=1).iterrows():
        if train_len > c:
            train_l.append(name)
            c += i['author']
        else:
            test_l.append(name)
    return data.loc[data['repo'].isin(train_l)], data.loc[data['repo'].isin(test_l)]


train_ds, test_ds = train_test_by_repo(data, TRAIN_TEST_SPLIT)


# %%
data.head()


# %%
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# %%
def process_elem(data_batch_i):
    sentence_line =  tokenizer(data_batch_i['body'], return_tensors='pt', padding='max_length', truncation=True)
    sentence_line1 = tokenizer(data_batch_i['body'], padding='max_length', truncation=True,  return_offsets_mapping=True, return_length=True)
    args = get_names(data_batch_i['body'])
    args = offset2ind(args, sentence_line1)
    labels = dict(zip(eval(data_batch_i['arg_names']), data_batch_i['labels']))
    ids = torch.zeros_like(sentence_line['input_ids'])
    for i in args:
        ids[0][i[1]]=labels[i[0]]
    return sentence_line, ids


def offset2ind(args, tokens):
    def find(tok, lis):
        r = []
        for i in lis:
            if i[0]>=tok[1][0] and i[1]<=tok[1][1]:
                r.append(i)
                return
        b = [lis.index(i) for i in r]
        return b

    return [(i[0], find(i,tokens['offset_mapping'])) for i in args]


# %%
from torch.utils.data import Dataset, DataLoader


class DataDataset(Dataset):

    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_batch = self.data.iloc[idx, :]
        full_sentence, ids = process_elem(data_batch)
        return {'input_ids': full_sentence['input_ids'].squeeze().to(device),
                    'attention_mask':full_sentence['attention_mask'].squeeze().to(device),
                    'input_mask': (ids > 0).squeeze().to(device),
                    'ids': ids.squeeze().to(device)}


# %%
train = DataLoader(DataDataset(train_ds), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)


# %%
class Model(torch.nn.Module):
    def __init__(self, bert, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.bert = bert
        self.dense = nn.Linear(768, out_dim)

    def forward(self, a):
        
        emb = self.bert(a['input_ids'], attention_mask=a['attention_mask'])['last_hidden_state']
        out = self.dense(emb)
        mask = a['input_mask'].unsqueeze(-1).expand(out.size())
        masked = torch.masked_select(out, mask).reshape(len(torch.masked_select(a['ids'], a['input_mask'])),self.out_dim)
        return masked


model = Model(bert, len(enc.classes_))
model.to(device)
print()


# %%
for param in model.bert.parameters():
    param.requires_grad = False


# %%
opti = torch.optim.Adam(model.parameters(), lr = 2e-5)
pbar = tqdm(total=len(train))
losses = []
accuracy = []
for i,a in enumerate(train):
    out = model.forward(a)
    labels = torch.masked_select(a['ids'], a['input_mask'])
    loss = F.cross_entropy(out, labels)
    if torch.isnan(loss):
        # print(a)
        pass
    else:
        accuracy.append(sum(torch.argmax(F.softmax(out), dim=1) == labels).detach()/len(labels))
        losses.append(loss.detach())
    loss.backward()
    opti.step()
    if i % 20 ==0:
        pbar.set_description(f"Loss : { sum(losses)/len(losses)}, acc: {sum(accuracy)/len(accuracy)}")
    pbar.update(1)
pbar.close()


# %%



# %%
test = DataLoader(DataDataset(test_ds), batch_size=1, num_workers=0)


# %%
pr_av = lambda x : sum(x)/len(x)


# %%
pbar = tqdm(total=len(test))
test_top_5s = []
test_accuracy = []
test_losses = []
for i,a in enumerate(test):
    out = model.forward(a)
    labels = torch.masked_select(a['ids'], a['input_mask'])
    loss = F.cross_entropy(out, labels)
    if torch.isnan(loss):
        # print(a)
        pass
    else:
        test_accuracy.append(sum(torch.argmax(F.softmax(out), dim=1) == labels).detach()/len(labels))
        test_losses.append(loss.detach())
        top5s = torch.topk(out, 5).indices
        correct_top5 = 0
        for i in range(len(labels)):
            if labels[i] in top5s[i]:
                correct_top5 += 1
        test_top_5s.append(correct_top5/len(labels))
    
    if i % 20 ==0:
        pbar.set_description(f"Loss : { pr_av(test_losses)}, acc: {pr_av(test_accuracy)}, top5s: {pr_av(test_top_5s)}")
    pbar.update(1)
pbar.close()


# %%



