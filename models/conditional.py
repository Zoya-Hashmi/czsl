## OUR EXPERIMENTS

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import MLP
import random

random.seed(2)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class conditional_module(nn.Module):
    def __init__(self,inp_dim,out_dim,hid_dim):
        super().__init__()
        self.w = MLP(2*inp_dim,out_dim,num_layers=2,dropout=True,norm=True,layers=[hid_dim],relu=False)
    def forward(self,a,o):
        xa = torch.sigmoid(self.w(torch.cat([a,o],1)))
        xa = xa*a
        return xa

class compnet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.attr_embdr = conditional_module(self.args.wemb_dim,self.args.wemb_dim,self.args.hid_dim)
        self.obj_embdr = conditional_module(self.args.wemb_dim,self.args.wemb_dim,self.args.hid_dim)
        self.comp = MLP(inp_dim=2*self.args.wemb_dim,out_dim=self.args.emb_dim,num_layers=2,dropout=False,norm=False,layers=[self.args.hid_dim],relu=False)

    def forward(self,a,o):
        xa,xo=None, None
        xa = self.attr_embdr(a,o)
        xo = self.obj_embdr(o,a)
        x = self.comp(torch.cat([xa,xo],1))
        return x,xa,xo

class Conditional(nn.Module):

    def __init__(self, dset, args):
        super().__init__()
        self.args = args
        self.dset = dset
        self.margin = args.margin

        # precompute validation pairs
        attrs, objs = zip(*self.dset.pairs)
        attrs = [dset.attr2idx[attr] for attr in attrs]
        objs = [dset.obj2idx[obj] for obj in objs]
        self.val_attrs = torch.LongTensor(attrs).to(device)
        self.val_objs = torch.LongTensor(objs).to(device)

        attrs, objs = zip(*self.dset.train_pairs)
        attrs = [dset.attr2idx[attr] for attr in attrs]
        objs = [dset.obj2idx[obj] for obj in objs]
        self.train_attrs = torch.LongTensor(attrs).to(device)
        self.train_objs = torch.LongTensor(objs).to(device)

        embeddings = torch.load(self.args.graph_init)["embeddings"]
        self.attr_embds = embeddings[:len(self.dset.attrs),:].to(device)
        self.obj_embds = embeddings[len(self.dset.attrs):len(self.dset.attrs)+len(self.dset.objs),:].to(device)

        self.image_embedder = MLP(inp_dim=self.args.vemb_dim,out_dim=self.args.emb_dim,num_layers=self.args.nlayers,dropout=True,norm=True,layers=[self.args.hid_dim,self.args.hid_dim],relu=True) #####
        self.comp_embedder = compnet(self.args)

        self.train_forward = self.train_forward_ce
        self.val_forward = self.val_forward_dotpr
                
        if args.lambda_aux>0:
            self.obj_clf = nn.Linear(self.args.wemb_dim, len(dset.objs)) 
            self.attr_clf = nn.Linear(self.args.wemb_dim, len(dset.attrs)) 
    
    def train_forward_ce(self, x):

        img, attrs, objs, pairs = x[0], x[1], x[2], x[3]

        img_feats = self.image_embedder(img)

        train_attrs,train_objs = self.attr_embds[self.train_attrs],self.obj_embds[self.train_objs]
        pair_embeds,attr_embed,obj_embed = self.compose(train_attrs, train_objs)
        
        pair_embeds = pair_embeds.permute(1,0)
        scores = torch.matmul(img_feats, F.normalize(pair_embeds))

        loss = F.cross_entropy(scores, pairs)
        
        if self.args.lambda_aux>0:
            obj_embed = obj_embed.index_select(0,pairs)
            attr_embed = attr_embed.index_select(0,pairs)
            obj_pred = self.obj_clf(obj_embed)
            attr_pred = self.attr_clf(attr_embed)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss += self.args.lambda_aux*loss_aux


        return loss, None

    def val_forward_dotpr(self,x):
        
        img = x[0]
        pairs = x[3]

        val_attrs = self.attr_embds[self.val_attrs]
        val_objs = self.obj_embds[self.val_objs]
        
        img_feats = self.image_embedder(img)
        pair_embeds,_,_ = self.compose(val_attrs, val_objs)

        score = torch.matmul(img_feats, F.normalize(pair_embeds.permute(1,0)))
     
        scores = {}
        for itr, pair in enumerate(self.dset.pairs):
            scores[pair] = score[:,self.dset.all_pair2idx[pair]]

        return None, scores
        
    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred

    def compose(self,attrs,objs):
        outputs = self.comp_embedder(attrs,objs)
        return outputs
