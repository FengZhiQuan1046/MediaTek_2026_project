"""LLM-SRec: frozen LLM retrieval with SASRec representation distillation.
Uses upstream llm4rec heads, output tokens, retrieval and matching objectives.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared_yelp import ROOT, module, load_shared, EvaluationWriter
sys.path.insert(0, str(HERE.parent/'SASRec'))
from model import SASRec
teacher_tools = module('teacher_tools', HERE.parent/'SASRec/train.py')
upstream = module('upstream_llm', HERE/'upstream/models/seqllm4rec.py')
# Keep every consumed item masked even when the teacher input is truncated.
_original_evaluation_sequences = teacher_tools.evaluation_sequences

def teacher_evaluation_sequences(data, users, split, maxlen, device):
    seq, _ = _original_evaluation_sequences(data, users, split, maxlen, device)
    histories = [list(data.train_by_user[u]) + ([data.valid_target[u]] if split == 'test' else []) for u in users]
    return seq, histories

teacher_tools.evaluation_sequences = teacher_evaluation_sequences



def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--data-path')
    p.add_argument('--cache-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--model-path',default='meta-llama/Llama-3.2-3B-Instruct')
    p.add_argument('--max-events',type=int)
    p.add_argument('--min-rating',type=float,default=4.)
    p.add_argument('--refresh-data-cache',action='store_true')
    p.add_argument('--maxlen',type=int,default=100)
    p.add_argument('--teacher-epochs',type=lambda value: min(7, int(value)),default=2)
    p.add_argument('--epochs',type=lambda value: min(8, int(value)),default=6)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--early-stopping-patience',type=int,default=10)
    p.add_argument('--seed',type=int,default=25252)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--learning-rate',type=float,default=1e-4)
    a=p.parse_args(); a.nn_parameter=False; a.token=False
    if a.teacher_epochs < 1 or a.epochs < 1:
        p.error('Teacher and LLM epochs must be positive')
    a.epochs = min(a.epochs, 8 - a.teacher_epochs)
    return a


class Retriever(nn.Module):
    def __init__(self,teacher,data,args):
        super().__init__()
        self.teacher=teacher.eval().requires_grad_(False)
        self.data=data; self.args=args
        self.llm=upstream.llm4rec(args.device,'llama-3b',args=args)
        self.item_emb_proj=nn.Sequential(nn.Linear(64,2048),nn.LayerNorm(2048),nn.LeakyReLU(),
                                        nn.Linear(2048,self.llm.llm_model.config.hidden_size))
        nn.init.xavier_normal_(self.item_emb_proj[0].weight)
        nn.init.xavier_normal_(self.item_emb_proj[3].weight)
        self.to(args.device)
        # Keep the frozen transformer deterministic, including during adapter training.
        self.llm.llm_model.eval()
        self.llm.llm_model.config.use_cache=False

    def train(self,mode=True):
        super().train(mode); self.teacher.eval(); self.llm.llm_model.eval(); return self

    def item_emb(self,ids):
        return self.item_emb_proj(self.teacher.item_embedding(torch.tensor(ids,device=self.args.device)+1))

    def user_samples(self,histories):
        texts=[]; embeddings=[]
        for history in histories:
            history=history[-10:]
            text=', '.join(f'Item No.{i+1}, "{self.data.item_texts[item][:256]}"[HistoryEmb]' for i,item in enumerate(history))
            texts.append('This user has made a series of purchases in the following order: '+text+
                         '. Based on this sequence of purchases, generate user representation token:[UserOut]')
            embeddings.append(self.item_emb(history))
        return texts,embeddings

    def item_texts(self,ids):
        return ['The item title and item embedding are as follows: "'+self.data.item_texts[i][:256]+
                '"[HistoryEmb], then generate item representation token:[ItemOut]' for i in ids]

    def vectors(self,texts,token,embeddings):
        tokens=self.llm.llm_tokenizer(texts,return_tensors='pt',padding=True,truncation=True,max_length=1024).to(self.args.device)
        inputs=self.llm.llm_model.get_input_embeddings()(tokens.input_ids)
        if token=='[UserOut]':
            inputs=self.llm.replace_out_token_all(tokens,inputs,token=[token,'[HistoryEmb]'],embs={'[HistoryEmb]':embeddings})
        else:
            inputs=self.llm.replace_out_token_all_infer(tokens,inputs,token=[token,'[HistoryEmb]'],embs={'[HistoryEmb]':embeddings})
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            out=self.llm.llm_model.model(inputs_embeds=inputs,attention_mask=tokens.attention_mask,use_cache=False).last_hidden_state
        positions=self.llm.get_embeddings(tokens,token)
        if any(len(x)!=1 for x in positions): raise RuntimeError('Output token lost during tokenization')
        pooled=torch.stack([out[i,x].mean(0) for i,x in enumerate(positions)]).float()
        return (self.llm.pred_user if token=='[UserOut]' else self.llm.pred_item)(pooled)

    def loss(self,histories,targets,rng):
        seq=torch.zeros((len(histories),self.args.maxlen),dtype=torch.long,device=self.args.device)
        candidates=[]
        for i,(history,target) in enumerate(zip(histories,targets)):
            h=history[-self.args.maxlen:];seq[i,-len(h):]=torch.tensor(h,device=seq.device)+1
            seen=set(history)|{target}; available=self.data.num_items-len(seen)
            if available<3: raise RuntimeError('Need three unseen negatives')
            neg=[]
            while len(neg)<3:
                item=rng.randrange(self.data.num_items)
                if item not in seen and item not in neg: neg.append(item)
            candidates.extend([target,*neg])
        with torch.no_grad(): log_emb=self.teacher.encode(seq)[:,-1]
        texts,embs=self.user_samples(histories)
        samples=dict(text_input=texts,log_emb=log_emb,candidates_pos=self.item_texts(candidates),
                     interact=embs,candidate_embs=self.item_emb(candidates))
        return self.llm(samples,mode=0)[0]


@torch.inference_mode()
def evaluate(model,data,split,args):
    model.eval();started=time.perf_counter();batch=args.eval_batch_size
    items=torch.cat([model.vectors(model.item_texts(list(range(i,min(i+batch,data.num_items)))),
                                  '[ItemOut]',model.item_emb(list(range(i,min(i+batch,data.num_items)))))
                     for i in tqdm(range(0,data.num_items,batch),desc='Catalog embeddings')])
    targets=data.valid_target if split=='valid' else data.test_target
    users=sorted(targets);totals={f'{m}@{k}':0. for m in ['recall','hit','ndcg'] for k in [5,10]}
    for start in tqdm(range(0,len(users),batch),desc=f'full-catalog {split}'):
        us=users[start:start+batch]
        histories=[list(data.train_by_user[u])+([data.valid_target[u]] if split=='test' else []) for u in us]
        texts,embs=model.user_samples(histories)
        scores=model.vectors(texts,'[UserOut]',embs)@items.T
        gold=torch.tensor([targets[u] for u in us],device=args.device)
        for row,h in enumerate(histories):
            seen=set(h)-{targets[us[row]]}
            scores[row,list(seen)]=-torch.inf
        ranks=(scores>=scores.gather(1,gold[:,None])).sum(1)
        for k in [5,10]:
            hits=ranks<=k
            totals[f'recall@{k}']+=hits.sum().item();totals[f'hit@{k}']+=hits.sum().item()
            totals[f'ndcg@{k}']+=torch.where(hits,1/torch.log2(ranks.float()+1),0.).sum().item()
    seconds=time.perf_counter()-started
    return {**{k:v/len(users) for k,v in totals.items()},'evaluated_users':len(users),'total_users':len(users),
            'users_per_second':len(users)/seconds,'scores_per_second':len(users)*data.num_items/seconds}


def state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()
            if k in {name for name,p in model.named_parameters() if p.requires_grad}}


def main():
    a=parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    logger=teacher_tools.configure_logging(out/time.strftime('train_%Y%m%d_%H%M%S.log'))
    (out/'config.json').write_text(json.dumps({**vars(a),'save_model_weights':False,
        'implementation':'upstream llm4rec loss and heads; shared SASRec teacher; full-catalog evaluation',
        'history_items':10,'timestamps':'omitted: release has no real timestamps'},indent=2))
    data,cache=load_shared(a,logger);writer=EvaluationWriter(out,a.dataset)
    history=[];best_test=None
    teacher=SASRec(data.num_items,a.maxlen,64,2,1,.2).to(a.device)
    optim=torch.optim.Adam(teacher.parameters(),lr=1e-3);rng=random.Random(a.seed)
    best=-math.inf;best_state=None;stale=0
    steps=max(1,len(data.train_by_user)//128)
    for epoch in range(1,a.teacher_epochs+1):
        teacher.train();total=0
        for _ in tqdm(range(steps),desc=f'SASRec teacher {epoch}'):
            seq,pos,neg=teacher_tools.sample_training_batch(data,128,a.maxlen,rng,a.device)
            pl,nl=teacher(seq,pos,neg);mask=pos.ne(0)
            loss=F.binary_cross_entropy_with_logits(pl[mask],torch.ones_like(pl[mask]))+F.binary_cross_entropy_with_logits(nl[mask],torch.zeros_like(nl[mask]))
            optim.zero_grad();loss.backward();optim.step();total+=loss.item()
        metrics=teacher_tools.evaluate(teacher,data,'valid',128,a.maxlen,a.device)
        metrics['total_users']=len(data.valid_target);writer.write('VALIDATION',metrics,'sasrec_teacher',epoch,epoch*steps)
        test=teacher_tools.evaluate(teacher,data,'test',128,a.maxlen,a.device)
        test['total_users']=len(data.test_target)
        writer.write('TEST',test,'sasrec_teacher',epoch,epoch*steps)
        history.append(dict(stage='sasrec_teacher',epoch=epoch,valid=metrics,test=test))
        (out/'metrics.json').write_text(json.dumps(dict(dataset=a.dataset,status='running',history=history),indent=2))
        logger.info('teacher epoch=%s loss=%s valid=%s test=%s',epoch,total/steps,metrics,test)
        if metrics['ndcg@10']>best:
            best=metrics['ndcg@10'];best_state=copy.deepcopy(teacher.state_dict());stale=0
        else: stale+=1
        if stale>=a.early_stopping_patience:break
    teacher.load_state_dict(best_state);del optim,best_state
    model=Retriever(teacher,data,a)
    optim=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=a.learning_rate)
    eligible=[u for u,h in data.train_by_user.items() if len(h)>1]
    steps=max(1,math.ceil(len(eligible)/a.batch_size));best=-math.inf;best_state=None;stale=0;best_epoch=0
    for epoch in range(1,a.epochs+1):
        model.train();rng.shuffle(eligible);total=0
        for start in tqdm(range(0,len(eligible),a.batch_size),desc=f'LLM-SRec {epoch}'):
            us=eligible[start:start+a.batch_size]
            if len(us)==1:us=us+[rng.choice(eligible)] # upstream uniformity needs >=2 users
            histories=[data.train_by_user[u][:-1] for u in us];targets=[data.train_by_user[u][-1] for u in us]
            optim.zero_grad();loss=model.loss(histories,targets,rng)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite LLM-SRec loss')
            loss.backward();optim.step();total+=loss.item()
        metrics=evaluate(model,data,'valid',a);writer.write('VALIDATION',metrics,'recommender',epoch,epoch*steps)
        test=evaluate(model,data,'test',a);writer.write('TEST',test,'recommender',epoch,epoch*steps)
        history.append(dict(stage='recommender',epoch=epoch,valid=metrics,test=test))
        logger.info('LLM-SRec epoch=%s loss=%s valid=%s test=%s',epoch,total/steps,metrics,test)
        if metrics['ndcg@10']>best:
            best=metrics['ndcg@10'];best_state=state(model);stale=0;best_epoch=epoch;best_test=test
        else:stale+=1
        (out/'metrics.json').write_text(json.dumps(dict(dataset=a.dataset,status='running',best_epoch=best_epoch,test=best_test,history=history),indent=2))
        if stale>=a.early_stopping_patience:break
    model.load_state_dict(best_state,strict=False)
    metrics=best_test;writer.write('FINAL TEST',metrics,'best_validation_model',best_epoch,best_epoch*steps)
    (out/'metrics.json').write_text(json.dumps(dict(dataset=a.dataset,status="completed",best_epoch=best_epoch,test=metrics,history=history),indent=2))
    logger.info('FINAL TEST %s',metrics)
if __name__=='__main__':main()
