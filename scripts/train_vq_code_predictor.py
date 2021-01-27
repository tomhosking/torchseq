# MLP code prediction

import argparse, json, os

parser = argparse.ArgumentParser(
    description="MLP code prediction trainer",
)
parser.add_argument(
 "--data_dir", type=str,  default='./data/', help="Path to data folder"
)
parser.add_argument(
 "--model_path", type=str,  default='./runs/sep_ae/20201230_132811_vae_wa_6h_quantized_256_16qh_chunk-drop30/', help="Path to model folder"
)
parser.add_argument(
 "--output_path", type=str,  default='./runs/mlpcodepredictor/', help="Path to output folder"
)

parser.add_argument(
 "--dataset", type=str,  default='wikianswers', help="Which dataset?"
)

parser.add_argument("--train", action="store_true", help="Train mode")
parser.add_argument("--eval", action="store_true", help="Eval mode")
parser.add_argument("--test", action="store_true", help="Eval on test")

parser.add_argument(
 "--lr", type=float, default=1e-4
)
parser.add_argument(
 "--bsz", type=int, default=1024
)
parser.add_argument(
 "--codebook_size", type=int, default=0
)
parser.add_argument(
 "--hidden_dim", type=int, default=768*4
)
parser.add_argument(
 "--num_steps", type=int, default=30001
)



args = parser.parse_args()

if args.dataset == 'wikianswers':
    dataset_all = 'wikianswers-para-allqs'
    dataset_clusters = 'wikianswers-pp'
    dataset_geneval = 'wikianswers-para-splitforgeneval'
    dataset_mlppredict = 'wikianswers-para-exemplarmlppredict'
elif args.dataset == 'qqp':
    dataset_all = 'qqp-allqs'
    dataset_clusters = 'qqp-clusters'
    dataset_geneval = 'qqp-splitforgeneval'
    dataset_mlppredict = 'qqp-exemplarmlppredict'
    

import torch
from torch.autograd import Variable
from tqdm import tqdm

from torchseq.utils.functions import onehot
from torchseq.utils.seed import set_seed

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_heads):
        super(MLPClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, input_dim*num_heads)
        self.linear2 = torch.nn.Linear(input_dim*num_heads, input_dim*num_heads)
        self.linear3 = torch.nn.Linear(input_dim*num_heads, output_dim*num_heads)

        self.drop1 = torch.nn.Dropout(p=0.2)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.num_heads = num_heads
        self.output_dim = output_dim

    def forward(self, x):
        outputs = self.drop1(torch.nn.functional.relu(self.linear(x)))
        outputs = self.drop2(torch.nn.functional.relu(self.linear2(outputs)))
        outputs = self.linear3(outputs)
        return outputs.reshape(-1, self.num_heads, self.output_dim)



os.makedirs(args.output_path, exist_ok=True)
with open(args.output_path + '/config.json', 'w') as f:
    json.dump(vars(args), f)


import numpy as np
import jsonlines, os

# Load encodings, data

MODEL_PATH = args.model_path

if not os.path.exists(MODEL_PATH+f'/sep_encoding_1_train.npy') or not os.path.exists(MODEL_PATH+f'/sep_encoding_1_dev.npy') or not os.path.exists(MODEL_PATH+f'/sep_encoding_1_test.npy'):
    # generate encodings
    print('Encoding cache not found - generating...')

    import json, torch, jsonlines
    from tqdm import tqdm
    import numpy as np

    from torchseq.agents.para_agent import ParaphraseAgent
    from torchseq.datasets.json_loader import JsonDataLoader

    from torchseq.utils.config import Config

    with open(MODEL_PATH + "/config.json") as f:
        cfg_dict = json.load(f)
        # cfg_dict["task"] = "autoencoder"
        cfg_dict["env"]["data_path"] = args.data_dir
        cfg_dict["eval"]["sample_outputs"] = False
        cfg_dict["training"]['batch_size'] = 24
        cfg_dict["eval"]['eval_batch_size'] = 24
        cfg_dict["training"]["dataset"] = 'json'
        cfg_dict["training"]["shuffle_data"] = False
        cfg_dict['json_dataset'] = {
            "path": dataset_all,
            "field_map": [
                {
                    "type": "copy",
                    "from": "q",
                    "to": "s2"
                },
                {
                    "type": "copy",
                    "from": "q",
                    "to": "s1"
                }
            ]
        }
        cfg_dict["bottleneck"]["prior_var_weight"] = 0.0

        config = Config(cfg_dict)


    checkpoint_path = MODEL_PATH

    data_loader = JsonDataLoader(config)

    instance = ParaphraseAgent(config=config, run_id=None, output_path="./runs/parademo/", silent=False, verbose=False)

    if os.path.exists(os.path.join(MODEL_PATH, "orig_model.txt")):
        with open(os.path.join(MODEL_PATH, "orig_model.txt")) as f:
            chkpt_pth = f.readlines()[0]
        checkpoint_path = chkpt_pth
    else:
        checkpoint_path = os.path.join(MODEL_PATH, "model", "checkpoint.pt")
    instance.load_checkpoint(checkpoint_path)
    instance.model.eval()

    # Train
    if not os.path.exists(MODEL_PATH+f'/sep_encoding_1_train.npy'):
        _, _, _, memory_train = instance.inference(data_loader.train_loader, memory_keys_to_return=['sep_encoding_1', 'sep_encoding_2','vq_codes'])

        torch.cuda.empty_cache()

        for mem_key in ['sep_encoding_1', 'sep_encoding_2','vq_codes']:
            np.save(MODEL_PATH+f'/{mem_key}_train.npy', memory_train[mem_key])
    
    # Dev
    if not os.path.exists(MODEL_PATH+f'/sep_encoding_1_dev.npy'):
        _, _, _, memory_dev = instance.inference(data_loader.valid_loader, memory_keys_to_return=['sep_encoding_1', 'sep_encoding_2','vq_codes'])

        torch.cuda.empty_cache()

        for mem_key in ['sep_encoding_1', 'sep_encoding_2','vq_codes']:
            np.save(MODEL_PATH+f'/{mem_key}_dev.npy', memory_dev[mem_key])
    
    # Test
    if not os.path.exists(MODEL_PATH+f'/sep_encoding_1_test.npy'):
        _, _, _, memory_test = instance.inference(data_loader.test_loader, memory_keys_to_return=['sep_encoding_1', 'sep_encoding_2','vq_codes'])

        torch.cuda.empty_cache()

        for mem_key in ['sep_encoding_1', 'sep_encoding_2','vq_codes']:
            np.save(MODEL_PATH+f'/{mem_key}_test.npy', memory_test[mem_key])

    del instance
    del data_loader

    torch.cuda.empty_cache()

    print('Encoding cache built')

# Now actually load the encodings

print('Loading encodings, data')

memory_train = {}
memory_dev = {}
memory_test = {}

for mem_key in ['sep_encoding_1', 'sep_encoding_2', 'vq_codes']:
    memory_train[mem_key] = np.load(MODEL_PATH+f'/{mem_key}_train.npy')
    
    if args.test:
        memory_test[mem_key] = np.load(MODEL_PATH+f'/{mem_key}_test.npy')
    else:
        memory_dev[mem_key] = np.load(MODEL_PATH+f'/{mem_key}_dev.npy')
    
with jsonlines.open(os.path.join(args.data_dir, dataset_clusters, "train.jsonl")) as f:
    train_qs = [row for row in f]
train_cluster_ixs = []
ix = 0
for cix, cluster in enumerate(train_qs):
    clen = len(cluster['qs'])
    for i in range(clen):
        cluster_ixs = list(range(ix, ix+clen))
        # if args.dataset != 'qqp':
        cluster_ixs.remove(ix + i)
        train_cluster_ixs.append(cluster_ixs)
    ix += clen
    
with jsonlines.open(os.path.join(args.data_dir, dataset_clusters, "dev.jsonl")) as f:
    dev_qs = [row for row in f]
dev_cluster_ixs = []
ix = 0
for cix, cluster in enumerate(dev_qs):
    clen = len(cluster['qs'])
    for i in range(clen):
        cluster_ixs = list(range(ix, ix+clen))
        # if args.dataset != 'qqp':
        cluster_ixs.remove(ix + i)
        dev_cluster_ixs.append(cluster_ixs)
    ix += clen

import sys, gc
gc.collect()
# print('mem train', sum([x.nbytes for x in memory_train.values()])/1024**2)
# print('mem dev', sum([x.nbytes for x in memory_dev.values()])/1024**2)
# print('mem test', sum([x.nbytes for x in memory_test.values()])/1024**2)
# print('qs train', sys.getsizeof(train_qs)/1024**2)
# print('qs dev', sys.getsizeof(dev_qs)/1024**2)
# print('clusters train', sys.getsizeof(train_cluster_ixs)/1024**2)
# print('clusters dev', sys.getsizeof(dev_cluster_ixs)/1024**2)

print('Data and encodings loaded')

# from guppy import hpy; 
# h=hpy()
# h.heap()
    

# Prepare datasets

print('Prepping dataset')
h_ix = 0

X = np.concatenate([memory_train['sep_encoding_1'][:, 0, :], memory_train['sep_encoding_2'][:, 0, :]], axis=1)
y = memory_train['vq_codes'][:, :, 0]

# print(y[:10, :])
# print(len(train_qs))
# print(X.shape)
# print(len(train_cluster_ixs))

# X_train_ixs = []
# y_train_ixs = []
# for src_ix, cluster in enumerate(train_cluster_ixs):
#     for tgt_ix in cluster:
#         X_train_ixs.append(src_ix)
#         y_train_ixs.append(tgt_ix)
        

# X_dev_ixs = []
# y_dev_ixs = []
# for src_ix, cluster in enumerate(dev_cluster_ixs[:1000]):
#     for tgt_ix in cluster:
#         X_dev_ixs.append(src_ix)
#         y_dev_ixs.append(tgt_ix)


if args.test:
    # X_dev = memory_dev['sep_encoding_1'][:, 0, :]
    X_test = np.concatenate([memory_test['sep_encoding_1'][:, 0, :], memory_test['sep_encoding_2'][:, 0, :]], axis=1)
    y_test = memory_test['vq_codes'][:, :, 0]
else:

    # X_dev = memory_dev['sep_encoding_1'][:, 0, :]
    X_dev = np.concatenate([memory_dev['sep_encoding_1'][:, 0, :], memory_dev['sep_encoding_2'][:, 0, :]], axis=1)
    y_dev = memory_dev['vq_codes'][:, :, 0]
print('Datasets prepped')

# Train the model


    
batch_size = args.bsz
NUM_STEPS = args.num_steps
NUM_HEADS = 4

input_dim = 768 * 4//4
output_dim = args.codebook_size
hidden_dim = args.hidden_dim
lr_rate = args.lr

set_seed(123)

model = MLPClassifier(input_dim, output_dim, hidden_dim, NUM_HEADS).cuda()

if args.train:

    print('Training model...')

    criterion = torch.nn.CrossEntropyLoss().cuda() # computes softmax and then the cross entropy

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    rand_ixs = np.random.randint(0, high=len(train_cluster_ixs), size=(NUM_STEPS, batch_size))

    best_acc = 0

    for iter in tqdm(range(NUM_STEPS)):
    #     batch_ixs = np.random.choice(len(train_cluster_ixs), size=batch_size)

        model.train()

        batch_ixs = rand_ixs[iter,:]
        
        inputs = Variable(torch.tensor([X[ix] for ix in batch_ixs])).cuda()
        
        # print([len(train_cluster_ixs[cix]) for cix in batch_ixs])

        tgt = torch.where(torch.cat([torch.sum(torch.cat([onehot(torch.tensor(y[ix]), N=output_dim).unsqueeze(0) for ix in train_cluster_ixs[cix]], dim=0), dim=0, keepdims=True) for cix in batch_ixs], dim=0) > 0, 1, 0).cuda()
        
    #     tgt = Variable(tgt).cuda()
        

        optimizer.zero_grad()
        outputs = model(inputs)
        

    #     loss = criterion(outputs, labels)
        loss = torch.sum(-1 * torch.nn.functional.log_softmax(outputs, dim=-1) * tgt/tgt.sum(dim=-1, keepdims=True), dim=-1).mean()  #
        loss.backward()
        optimizer.step()
        
        if iter%1000==0:
            model.eval()
            # calculate Accuracy
            correct = 0
            all_acc = 0
            head_acc = [0] * NUM_HEADS
            total = 0
            for x_ix, cluster in enumerate(train_cluster_ixs[:10000]):
                inputs = Variable(torch.tensor([X[x_ix]])).cuda()
                labels = cluster
                outputs = model(inputs)
                predicted = torch.argmax(outputs.data, -1).cpu()
                total+= inputs.size(0)
                # for gpu, bring the predicted and labels back to cpu fro python operations to work
    #             print(predicted, [y[ix] for ix in cluster])
                
                all_correct = True
                for h_ix in range(NUM_HEADS):
                    this_corr = (predicted[0, h_ix] in [y[ix, h_ix] for ix in cluster])
                    correct+= 1.0 * this_corr
                    head_acc[h_ix] += 1.0 * this_corr
                    all_correct  = all_correct & this_corr
                all_acc += 1.0 * all_correct 
            accuracy = 100 * correct/(total*NUM_HEADS)
            head_acc = [100*x/total for x in head_acc]
            all_accuracy = 100 * all_acc/total
            if accuracy > best_acc:
                print('Saving...')
                torch.save(model.state_dict(), args.output_path+'/code_predict.pt')
                best_acc = accuracy
                metrics = {
                    'acc': accuracy, 
                    'full_acc': all_accuracy, 
                    'head_acc': head_acc
                }
                with open(args.output_path + '/metrics.json', 'w') as f:
                    json.dump(metrics, f)
            print("Iteration: {}. Loss: {}. Recall: {}. All Recall {}. PerHead Recall {}".format(iter, loss.item(), accuracy, all_accuracy, head_acc))


    print('Training complete')



# Run inference
if args.eval or args.test:

    split = 'test' if args.test else 'dev'

    print('Generating exemplars')

    import jsonlines, os, copy
    from tqdm import tqdm

    NUM_HEADS = 16
    NUM_TEMPL_HEADS = 4

    model.load_state_dict(torch.load(args.output_path+'/code_predict.pt'))
    model.eval()



    with jsonlines.open(os.path.join(args.data_dir, f"{dataset_geneval}/{split}.jsonl")) as f:
        rows = [row for row in f]
        
    q_to_ix = {}
    ix = 0
    with jsonlines.open(os.path.join(args.data_dir, f"{dataset_clusters}/{split}.jsonl")) as f:
        dev_qs = [row for row in f]
    for cix, cluster in enumerate(dev_qs):
        for q in cluster['qs']:
            q_to_ix[q] = ix
            ix += 1
        
            
    miss = 0
    # os.makedirs(args.data_dir + '/wikianswers-para-exemplarmlppredict', exist_ok=True)
    # with jsonlines.open(args.data_dir + '/wikianswers-para-exemplarmlppredict/dev.jsonl', 'w') as f:

    #     for ix, row in enumerate(tqdm(rows)):
    #         query_ix = q_to_ix[row['sem_input']]
    #         tgt_codes = [0] * (NUM_HEADS - NUM_TEMPL_HEADS)

    #         inputs = Variable(torch.tensor([X_dev[query_ix]])).cuda()

    #         outputs = model(inputs)
    #         predicted = torch.argmax(outputs.data, -1).cpu()
            
    #         gold = y_dev[ix]
            
    # #         print(predicted, gold)
            
            
    #         for h_ix in range(NUM_TEMPL_HEADS):
    #             tgt_codes.append(predicted[0, h_ix].item())

    #         this_row = copy.copy(row)
    #         this_row['vq_codes'] = tgt_codes
    #         f.write(this_row)

    X_src = X_test if args.test else X_dev

    os.makedirs(args.data_dir + '/' + dataset_mlppredict, exist_ok=True)
    with jsonlines.open(args.data_dir + '/' + dataset_mlppredict +f'/{split}.jsonl', 'w') as f:

        for ix, row in enumerate(tqdm(rows)):
            query_ix = q_to_ix[row['sem_input']]
            tgt_codes = [0] * (NUM_HEADS - NUM_TEMPL_HEADS)

            inputs = Variable(torch.tensor([X_src[query_ix]])).cuda()

            outputs = model(inputs)[0]
            probs, predicted = torch.topk(torch.softmax(outputs, -1), 3 -1)

    #         print(predicted.shape, probs.shape)
        #     break

            joint_probs = [([], 0)]
            for h_ix in range(NUM_TEMPL_HEADS):
                new_hypotheses = []
                for i, (combo, prob) in enumerate(joint_probs):
                    for k in range(2):
                        new_hyp = [copy.copy(combo), prob]
                        new_hyp[0].append(predicted[h_ix, k].item())
                        new_hyp[1] += torch.log(probs[h_ix, k]).item()

                        new_hypotheses.append(new_hyp)

                joint_probs = new_hypotheses
                joint_probs = sorted(joint_probs, key=lambda x: x[1], reverse=True)[:3]
            pred_codes = [tgt_codes + x[0] for x in sorted(joint_probs, key=lambda x: x[1], reverse=True)[:2]]
            
            # pred_codes = predicted.transpose(1,0).tolist()
            # pred_codes = [tgt_codes + codes for codes in pred_codes]
            # print(pred_codes)
            # exit()
    

        #     for h_ix in range(NUM_TEMPL_HEADS):
        #         tgt_codes.append(predicted[0, h_ix].item())
            for codes in pred_codes:
                this_row = copy.copy(row)
                this_row['vq_codes'] = codes
                f.write(this_row)
