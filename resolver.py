import glob
import os
import re
import json
import gzip
import numpy as np
import itertools as it
import pickle
import argparse
import time
import pysam
import multiprocessing as mp
from sklearn.experimental import enable_iterative_imputer
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import utils

simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=UserWarning)
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

#make a general purpose weighted f_1
def weighted_f1(prec,rec,p_w=1.0):
    return (p_w+1)/(p_w/prec+1/rec)

#fusion methods::::::::::::::::::::::::::::::::::::
def pos_to_wcu(pos,sid,label=None):
    return [[pos[i][0],pos[i][1],pos[i][2],{label:set([(sid,i)])}] for i in range(len(pos))]

def weight_graph(C):
    G,X = {},[]
    c_i,i_c = group_indecies(C)
    if len(i_c)>0: #load all unique verticies
        for c in sorted(list(C.keys())): #for each classifier
            for i in range(len(C[c])): #for every entry in every classifier
                for j in range(2):     #for the start and stop point in the pos (range)
                    if C[c][i][j] in G:
                        if c in G[C[c][i][j]]:
                            if j in G[C[c][i][j]][c]: G[C[c][i][j]][c][j] += [[i+c_i[c],C[c][i][2]]]
                            else:                     G[C[c][i][j]][c][j]  = [[i+c_i[c],C[c][i][2]]]
                        else:                         G[C[c][i][j]][c]  = {j:[[i+c_i[c],C[c][i][2]]]}
                    else:                             G[C[c][i][j]]  = {c:{j:[[i+c_i[c],C[c][i][2]]]}}
            X += C[c] #load the entry into X which will then get sorted by position
        last = sorted(list(G.keys()))[-1]+2          #get last x2 value in G
        G[last] = {(None,):{None:[None,None]}} #to pad out a terminal vertex
        #check all indecies are paired
        V = {}
        for i in sorted(G)[:-1]:
            for c in G[i]:
                for e in G[i][c]:
                    for l in G[i][c][e]:
                        if l[0] in V: V[l[0]] += 1
                        else:         V[l[0]]  = 1
        #print('all vertices and edges added : %s'%all([V[i]==2 for i in V]))
    return X,G,c_i,i_c

def group_indecies(C):
    c_i,i_c,i = {},{},0
    for g in sorted(C.keys()):
        c_i[g]=i
        for j in range(i,i+len(C[g])): i_c[j]=g
        i+=len(C[g])
    return c_i,i_c

def get_x_i(A):
    x_i = set([])
    for c in A:
        for e in A[c]:
            for l in A[c][e]: x_i.add(l[0])
    return sorted(list(x_i))

def del_edges(A):
    k = list(A.keys())
    for i in range(len(k)): #take value 1 keys for each class
        if 1 in A[k[i]]:
            while len(A[k[i]][1]) > 0:
                A[k[i]][0].remove(A[k[i]][1][0])
                A[k[i]][1].remove(A[k[i]][1][0])
            if len(A[k[i]][0])<1: A[k[i]].pop(0)
            if len(A[k[i]][1])<1: A[k[i]].pop(1)
        if len(A[k[i]])<1: A.pop(k[i])

def add_edges(A,B):
    for k in B:
        if k in A:
            for e in B[k]:
                if e in A[k]: A[k][e] += [l for l in B[k][e]]
                else:         A[k][e]  = [l for l in B[k][e]]
        else:                 A[k]  = {e:[l for l in B[k][e]] for e in B[k]}

def scan_graph(G,filter=None): #while offer additional weighting corrections
    X,C,c_i,i_c = G
    P,Q,V,A,B,D = [],[],[],{},{},set([]) #A is alive edge stack, B is the current edge set
    if len(X)>0:
        V = sorted(list(C.keys()))  #V is sorted vertices, where disjoint => |V|-1 = |X|*2
        for i in range(0,len(V)-1):   #scan i-1,i,i+1 up to padding
            B = C[V[i]]               #get edges for v[i+1]
            D = {m for l in [C[V[i]][k] for k in C[V[i]]] for m in l}  #check edge direction for i
            if len(A) <= 0:         #[1] len(a) <= 0 (f has 0 edge)   #section starting, start new  p+=[]
                add_edges(A,B)
                x_i =  get_x_i(A)
                P += [[V[i],V[i],0.0,join_idx(X,x_i,3)]]
                if D == set([0,1]):   #check for singles
                    del_edges(A)      #clean the singles
                    if len(A)>0:      #with new sections that are not singles
                        x_i =  get_x_i(A)
                        P += [[V[i]+1,V[i]+1,0.0,join_idx(X,x_i,3)]]
            else:
                if D == set([0]):   #[2] len(a) > 0 and f has 0 edge  #close subsection, start new  p[-1],p+=[]
                    x_i =  get_x_i(A)
                    P[-1][1] = V[i]-1
                    P[-1][3] = merge_idx(P[-1][3],join_idx(X,x_i,3))
                    del_edges(A)      #clean the closed edges
                    add_edges(A,B)    #start the new section
                    x_i =  get_x_i(A)
                    P += [[V[i],V[i],0.0,join_idx(X,x_i,3)]]
                if D == set([0,1]): #[3] len(a) > 0 and f has 0 and 1 #close subsection, set single p[-1],p+=[]
                    x_i =  get_x_i(A)
                    P[-1][1] = V[i]-1 #close the last open section
                    P[-1][3] = merge_idx(P[-1][3],join_idx(X,x_i,3))
                    add_edges(A,B)
                    x_i =  get_x_i(A)
                    P += [[V[i],V[i],0.0,join_idx(X,x_i,3)]]
                    del_edges(A)
                    if len(A)>0:
                        x_i =  get_x_i(A)
                        P += [[V[i]+1,V[i]+1,0.0,join_idx(X,x_i,3)]]
                if D == set([1]):   #[4] len(a) > 0 and f has 1       #section closing,  fix last   p[-1]
                    x_i =  get_x_i(A)
                    P[-1][1] = V[i]   #close and clean the last section
                    P[-1][3] = merge_idx(P[-1][3],join_idx(X,x_i,3))
                    add_edges(A,B)
                    del_edges(A)
                    if len(A)>0:      #find any remaining open sections
                        x_i =  get_x_i(A)
                        P += [[V[i]+1,V[i]+1,0.0,join_idx(X,x_i,3)]]
    for p in P: #clean up dangling sections
        if p[1]<p[0]: P.remove(p)
    if filter is not None:
        for p in P:
            if p[2]>filter: Q += [p]
    return P

def join_idx(C,j,p):
    I = {}
    for idx in [C[i][p] for i in j]:
        for k in idx:
            for i in idx[k]:
                if k in I: I[k].add(i)
                else:      I[k] =  {i}
    return I

def merge_idx(idx1,idx2):
    I = {}
    for k in idx1:
        for i in idx1[k]:
            if k in I: I[k].add(i)
            else:      I[k] =  {i}
    for k in idx2:
        for i in idx2[k]:
            if k in I: I[k].add(i)
            else:      I[k] =  {i}
    return I
#fusion methods::::::::::::::::::::::::::::::::::::

#sampling methods++++++++++++++++++++++++++++++++++
#look for close samples like: NA19239, NA19238, NA19240 which are related
def group_sample_ids(samples,idx_prox=4):
    sms  = {}
    prefix_pat = re.compile('^[a-zA-Z]+')
    suffix_pat = re.compile('[a-zA-Z]+$')
    int_pat    = re.compile('[0-9]+')
    for _sm in sorted(samples):
        prefix,suffix,idx = '','',None
        sm = _sm.split('/')[-1]
        if prefix_pat.search(sm) is not None: prefix = prefix_pat.search(sm).group()
        if suffix_pat.search(sm) is not None: suffix = suffix_pat.search(sm).group()
        if int_pat.search(sm) is not None:     idx   = int(int_pat.search(sm).group())
        if prefix in sms:
            idxs = sorted(sms[prefix])
            close_idx = np.where(abs(np.array(idxs)-idx)<idx_prox)[0]
            if len(close_idx)<1:
                if idx in sms[prefix]:
                    if suffix in sms[prefix][idx]: sms[prefix][idx][suffix] += [_sm]
                    else:                          sms[prefix][idx][suffix]  = [_sm]
                else:                              sms[prefix][idx] = {suffix:[_sm]}
            else:
                c_idx = idxs[close_idx[0]]
                if suffix in sms[prefix][c_idx]: sms[prefix][c_idx][suffix] += [_sm]
                else:                            sms[prefix][c_idx][suffix]  = [_sm]
        else:                      sms[prefix] = {idx:{suffix:[_sm]}}
    G = []
    for prefix in sms:
        for idx in sms[prefix]:
            for suffix in sms[prefix][idx]:
                G += [{len(sms[prefix][idx][suffix]):sms[prefix][idx][suffix]}]
    return sorted(G,key=lambda x: (sorted(x)[0],x[sorted(x)[0]][0]))[::-1]

def group_mag(group):
    return np.array([sorted(g)[0] for g in group])

def reduce_subgroup(group,x):
    n_over = sum(group_mag(group))-x
    _min   = n_over
    for i in range(4*n_over): #give it a few tries
        ridx = np.random.choice(range(len(group)),1)[0] #randomindex
        l    = len(group[ridx][sorted(group[ridx])[0]]) #how many would be lost
        m    = sum(group_mag(group))                    #current size, x is target
        if abs((m-l)-x)<_min:        #keep track of the closest to the target we get
            _min = abs((m-l)-x)
            group.pop(ridx)
    return group

def remove_duplicates(group):
    D,G = [],{}
    for row in group:
        for path in row[sorted(row)[0]]:
            sm = path.split('/')[-1]
            if sm in G: G[sm] += [path]
            else:       G[sm]  = [path]
    for sm in G:
        if len(G[sm])>1: D += [G[sm][np.random.choice(range(len(G[sm])),1)[0]]]
        else:            D += G[sm]
    return sorted(D)

def group_paths(group):
    G = []
    for i in range(len(group)):
        k = sorted(group[i])[0]
        for path in group[i][k]: G += [path]
    return sorted(G)

def sample_from_group(group,w,weighted_p=False):
    sample_mag   = group_mag(group)   #how many samples per cluster
    n_samples    = sum(sample_mag)           #count total observations
    if weighted_p:
        group = sorted(np.random.choice(group,w,replace=False,p=sample_mag/n_samples),key=lambda x: (sorted(x)[0],x[sorted(x)[0]][0]))[::-1]
    else:
        group = sorted(np.random.choice(group,w,replace=False),key=lambda x: (sorted(x)[0],x[sorted(x)[0]][0]))[::-1]
    return group

def group_diff(super_group,sub_group):
    A,B,D = {},{},[]
    for i in range(len(super_group)):
        for path in super_group[i][sorted(super_group[i])[0]]: A[path] = i
    for i in range(len(sub_group)):
        for path in sub_group[i][sorted(sub_group[i])[0]]:     B[path] = i
    remove = set([])
    for a in A:
        for b in B:
            if a==b: remove.add(A[a])
    for i in range(len(super_group)):
        if i not in remove: D += [super_group[i]]
    return sorted(D,key=lambda x: (sorted(x)[0],x[sorted(x)[0]][0]))[::-1]

def partition_samples_seqs(samples,seqs,split=0.6,remove_dups=True,r_seed=None):
    if r_seed is not None: np.random.seed(r_seed)
    sample_group = group_sample_ids(samples) #group adacent/related sample_ids like NA19238,NA19239,NA19240...
    if remove_dups:                          #randomly select a duplicate if desired...
        sample_group = group_sample_ids(remove_duplicates(sample_group))
    sample_mag   = group_mag(sample_group)   #how many samples per cluster
    n_samples    = sum(sample_mag)           #count total observations
    w = int(round(n_samples*split))          #desired number of training
    y = int(round((n_samples-w)*split))      #desired number of test
    z = n_samples-(w+y)                      #number of validation is the remaining
    #training----------------------------------------------------------------------
    train_group   = sample_from_group(sample_group,w)
    train_group   = reduce_subgroup(train_group,w)
    train_samples = group_paths(train_group)
    #test--------------------------------------------------------------------------
    open_group    = group_diff(sample_group,train_group)
    test_group    = sample_from_group(open_group,y)
    test_group    = reduce_subgroup(test_group,y)
    test_samples  = group_paths(test_group)
    #validation---------------------------------------------------------------------
    valid_group   = group_diff(open_group,test_group)
    valid_group   = reduce_subgroup(valid_group,z)
    valid_samples = group_paths(valid_group)

    #seqs are easier----------------------------------------------------------------
    train_seqs = sorted(list(np.random.choice(seqs,int(len(seqs)*(1.0-split)),replace=False)),key=lambda x: x.zfill(255))
    open_seqs  = sorted(set(seqs).difference(set(train_seqs)))
    test_seqs  = sorted(list(np.random.choice(open_seqs,int(len(open_seqs)*(1.0-split)),replace=False)),key=lambda x: x.zfill(255))
    valid_seqs = sorted(set(seqs).difference(set(train_seqs).union(set(test_seqs))))
    return {'samples':[train_samples,test_samples,valid_samples],
            'seqs':[train_seqs,test_seqs,valid_seqs]}

#--------------------------------------------------
def overlap(C1,C2):
    if C1[0]<=C1[1]: a,b = C1[0],C1[1]
    else:            a,b = C1[1],C1[0]
    if C2[0]<=C2[1]: c,d = C2[0],C2[1]
    else:            c,d = C2[1],C2[0]
    i = abs(a-c)+abs(b-d)
    u = min((b-a+1)+(d-c+1),max(b,d)-min(a,b)+1)
    return max(0.0,(u-i)/u)

def get_intersection_idx(X,Y,over=0.5):
    A = set([])
    for i in range(len(X)):
        for j in range(len(Y)):
            if overlap(X[i],Y[j])>over:
                A.add((i,j))
    return sorted(A)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# create a fast || per sample/seq intersection=>r_over method
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def get_info_v(info,p):
        m = re.search(p,info)
        if m is None: v = None
        else:         v = info[m.end():].split(';')[0]
        return v

def read_fasta_chrom_pos(fasta_path,chrom,start,stop):
    ss =''
    with pysam.FastaFile(fasta_path) as f:
        ss = f.fetch(chrom,start,stop)
    return ss

def hgsv2tsv_to_vcf(tsv_path,ref_path,vcf_path,default='1'):
    raw,data,columns = [],[],['CHROM','POS','ID','END','SVTYPE','SVLEN','GT','MERGE_SAMPLES','CALLERSET_LIST']
    if tsv_path.endswith('.gz'):
        with gzip.GzipFile(tsv_path,'rb') as f:
            raw = [line.decode('utf-8').replace('\n','').split('\t') for line in f.readlines()]
    else:
        with open(tsv_path,'r') as f:
            raw = [line.replace('\n','').split('\t') for line in f.readlines()]
    if len(raw)>0:
        header = [e.replace('#','') for e in raw[0]]
        c_idx  = {header[i]:i for i in range(len(header))}
        for row in raw[1:]:
            data += [[row[c_idx[c]] for c in columns]]
    #construct GT fields..............................
    sms = set([])
    for row in data:
        for sm in row[7].split(','): sms.add(sm)
    sms   = sorted(sms)
    s_idx = {sms[i]:i for i in range(len(sms))}
    for i in range(len(data)):
        gt = ['0/0' for sm in sms]
        data[i][6] = data[i][6].replace('.',default)
        for sm in data[i][7].split(','):
            gt[s_idx[sm]] = data[i][6]
        data[i][6] = gt
    V = []
    preamble = """##fileformat=VCFv4.2
    ##fileDate=20201127
    ##source=PAV (HGSVC2)
    ##reference=%s
    ##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
    ##INFO=<ID=END,Number=1,Type=Integer,Description="End coordinate of this variant">
    ##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
    ##INFO=<ID=CALLERSET_LIST,Number=.,Type=String,Description="HGSV2 Method Used">
    ##INFO=<ID=IMPRECISE,Number=0,Type=Flag,Description="Imprecise structural variation">
    ##ALT=<ID=INV,Description="Inversion">
    ##ALT=<ID=INS:MT,Description="Nuclear Mitochondrial Insertion">
    ##ALT=<ID=INS:ME:SVA,Description="Insertion of SVA element">
    ##ALT=<ID=INS:ME:LINE1,Description="Insertion of LINE1 element">
    ##ALT=<ID=INS:ME:ALU,Description="Insertion of ALU element">
    ##ALT=<ID=DUP,Description="Duplication">
    ##ALT=<ID=DEL,Description="Deletion">
    ##ALT=<ID=CNV,Description="Copy Number Polymorphism">
    ##ALT=<ID=CN9,Description="Copy number allele: 9 copies">
    ##ALT=<ID=CN8,Description="Copy number allele: 8 copies">
    ##ALT=<ID=CN7,Description="Copy number allele: 7 copies">
    ##ALT=<ID=CN6,Description="Copy number allele: 6 copies">
    ##ALT=<ID=CN5,Description="Copy number allele: 5 copies">
    ##ALT=<ID=CN4,Description="Copy number allele: 4 copies">
    ##ALT=<ID=CN3,Description="Copy number allele: 3 copies">
    ##ALT=<ID=CN2,Description="Copy number allele: 2 copies">
    ##ALT=<ID=CN1,Description="Copy number allele: 1 copy">
    ##ALT=<ID=CN0,Description="Copy number allele: 0 copies">
    ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
    """%(ref_path.split('/')[-1])
    vcf_header = ['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO','FORMAT']+sms
    for i in range(len(data)):
        info = 'SVTYPE=%s;SVLEN=%s;END=%s;CALLERSET_LIST=%s'%(data[i][4],data[i][5],data[i][3],data[i][8])
        ref  = read_fasta_chrom_pos(ref_path,data[i][0],int(data[i][1]),int(data[i][1])+1)
        alt  = '<%s>'%data[i][4]
        V += [data[i][0:3]+[ref,alt,'.','PASS',info,'GT']+data[i][6]]
    s = preamble.replace(' ','')+'\t'.join(vcf_header)+'\n'
    s += '\n'.join(['\t'.join(row) for row in V])+'\n'
    with open(vcf_path,'w') as f: f.write(s)
    return True

def vcf_kv(vcf_row):
    base = vcf_row.split('##')[-1]
    if type(base)==str:
        if base.startswith('#'):
            k,v,w = base,None,None
        elif len(base.split('='))>2:
            k = base.split('=')[0]
            v = base.split('=')[1].replace('<','')
            w = base.split('=')[2].split(',')[0]
        else:
            k,v,w = base.split('=')[0],base.split('=')[1].replace('<',''),None
    return k,v,w

def hgsv2_merge_insdel_inv_vcf(insdel_path,inv_path,vcf_out):
    with open(insdel_path,'r') as f:
        insdel_raw = [row.replace('\n','').split('\t') for row in f.readlines()]
        insdel_header ,insdel_data = [],[]
        for row in insdel_raw:
            if row[0].startswith('#'): insdel_header += [row]
            else:                      insdel_data   += [row]
        with open(inv_path,'r') as f:
            inv_raw    = [row.replace('\n','').split('\t') for row in f.readlines()]
            inv_header,inv_data = [],[]
            for row in inv_raw:
                if row[0].startswith('#'): inv_header += [row]
                else:                      inv_data   += [row]
            if insdel_header[-1][9:]==inv_header[-1][9:]:
                inv_patch = []
                for row in inv_header:
                    if row[0].startswith('##INFO') or row[0].startswith('##ALT') or row[0].startswith('##FORMAT'):
                        inv_patch += [row]
                I,D = {},{}
                for i in range(len(insdel_header)):
                    row = insdel_header[i]
                    k,v,w = vcf_kv(row[0])
                    if k in D:
                        if v in D[k]: D[k][v][w] = row
                        else:         D[k][v]    = {w:row}
                    else:             D[k]       = {v:{w:row}}
                for i in range(len(inv_patch)):
                    row = inv_patch[i]
                    k,v,w = vcf_kv(row[0])
                    if k in I:
                        if v in I[k]: I[k][v][w] = row
                        else:         I[k][v]    = {w:row}
                    else:             I[k]       = {v:{w:row}}

                for k in I:
                    if k in D:
                        for v in D[k]:
                            if v in I[k]:
                                for w in I[k][v]:
                                    D[k][v][w] = I[k][v][w]
                    else: D[k] = I[k]
                #build back up the header
                header,hs = [],['fileformat','fileDate','source','reference','contig','INFO','ALT','FORMAT','#CHROM']
                for k in hs:
                    if k=='#CHROM': header += ['\t'.join(D[k][None][None])]
                    elif k in D:
                        for v in D[k]:
                            for w in D[k][v]:
                                header += D[k][v][w]
                data = []
                for row in insdel_data: data += [row]
                for row in inv_data:    data += [row]
                data = sorted(data,key=lambda x: (x[0].zfill(255),int(x[1])))
                s  = '\n'.join(header)+'\n'
                s += '\n'.join(['\t'.join(row) for row in data])+'\n'
                with open(vcf_out,'w') as f: f.write(s)
    return True

def hgsv_sample_splitter(vcf_path,out_dir,add_chr=True,exp_ids=[1,2,3,4]):
    raw = []
    if vcf_path.endswith('.gz'):
        with gzip.GzipFile(vcf_path,'rb') as f: raw = [row.decode('utf8').replace('\n','').split('\t') for row in f.readlines()]
    else:
        with open(vcf_path,'r') as f:           raw = [row.replace('\n','').split('\t') for row in f.readlines()]
    header,sms = [],{}
    sample_pattern = re.compile('\ASAMPLE=|[;]SAMPLE=|[\s]SAMPLE=')
    type_pattern   = re.compile('\ASVTYPE=|[;]SVTYPE=|[\s]SVTYPE=')
    exp_pattern    = re.compile('\AEXPERIMENT=|[;]EXPERIMENT=|[\s]EXPERIMENT=')
    #exp_id= {1:BioNano (optimal map), 2:Illumina (Hiseq), 3: PacBio RSII, 4: Strandseq, 5: 1+2+3}
    for i in range(len(raw)):
        if raw[i][0].startswith('#'): header += [raw[i]]
        else:
            if add_chr and (not raw[i][0].startswith('chr')): raw[i][0] = 'chr'+raw[i][0]
            sm      = get_info_v(raw[i][7],sample_pattern)
            sv_type = get_info_v(raw[i][7],type_pattern)
            exp     = int(get_info_v(raw[i][7],exp_pattern))
            if exp in exp_ids:
                if sm not in sms: sms[sm] = {}
                if sv_type not in sms[sm]: sms[sm][sv_type] = []
                sms[sm][sv_type] += [raw[i]]
    for sm in sms:
        s,S = '\n'.join(['\t'.join(row) for row in header])+'\n',[]
        print('sample=%s------------'%sm)
        for sv_type in sorted(sms[sm]):
            print('sv=%s: %s'%(sv_type,len(sms[sm][sv_type])))
            S += sms[sm][sv_type]
        S = sorted(S,key=lambda x: (x[0].zfill(255),int(x[1])))
        s += '\n'.join(['\t'.join(row) for row in S])
        out_path = out_dir+'/%s_S0.vcf'%sm
        with open(out_path,'w') as f: f.write(s)

#universal reader algorithm from FusorSV or something else...
def vcf_reader(vcf_path,seqs=[str(x) for x in range(1,23)]+['X','Y'],
               sv_types=['DEL','DUP','INV','INS'],min_size=50,max_size=int(1e6),add_chr=True):
    header,raw = [],set([])
    alt_pattern    = re.compile('[^actgnACTGN,]')
    type_pattern   = re.compile('\ASVTYPE=|[;]SVTYPE=|[\s]SVTYPE=')
    simple_pattern = re.compile('\ASIMPLE_TYPE=|[;]SIMPLE_TYPE=|[\s]SIMPLE_TYPE=')
    end_pattern    = re.compile('\AEND=|[;]END=|[\s]END=')
    len_pattern    = re.compile('\ASVLEN=|[;]SVLEN=|[\s]SVLEN=')
    cnv_pattern    = re.compile('\AnatorRD=|[;]natorRD=|[\s]natorRD=')
    cons_pattern   = re.compile('\ACONSENSUS=|[;]CONSENSUS=|[\s]CONSENSUS=')
    icn_pattern    = re.compile('\AICN=|[;]ICN=|[\s]ICN=')
    with open(vcf_path,'r') as f: #raw = [row.replace('\n','').split('\t') for row in f.readlines()]
        for line in f:
            row = line.replace('\n','').split('\t')
            if line.startswith('#'): header += [row]
            elif row[1].isdigit(): #data row here--------------------------------------
                seq,start,ref,alt,filt,ins    = row[0],int(row[1]),row[3],row[4],row[6].upper(),None
                if add_chr and not seq.startswith('chr'): seq = 'chr%s'%seq
                if get_info_v(alt,alt_pattern) is not None:
                    alt = alt.replace('<','').replace('>','')
                svtype     = get_info_v(row[7],type_pattern)
                simpletype = get_info_v(row[7],simple_pattern)
                svlen      = get_info_v(row[7],len_pattern)
                end        = get_info_v(row[7],end_pattern)
                if simpletype is not None: svtype = simpletype
                if svtype is None:
                    if alt in sv_types: svtype = alt
                    else:
                        if len(ref)>len(alt):   svtype = 'DEL'
                        elif len(ref)<len(alt): svtype = 'INS'
                        else:                   svtype = 'MNV'
                if svlen is not None: svlen = abs(int(svlen))
                if end is None:
                    if svlen is None:
                        end = start+abs(len(ref)-len(alt))
                    else: end = start+svlen-1
                else: end = int(end)
                if svlen is None: svlen = max(1,abs(end-start)+1)
                if svlen==0:      svlen = max(1,abs(end-start)+1)
                if svtype=='CNV' and alt.find('CN')>-1:
                    if alt.split(',')[-1].split('CN')[-1].isdigit():
                        last_cn = int(alt.split(',')[-1].split('CN')[-1])
                        if last_cn>2: svtype = 'DUP'
                if svtype=='ALU' or svtype=='ME' or svtype=='MEI' or svtype =='INS':
                    svtype = 'INS' #extend this list...
                    #try to get it from the ALT or INFO fields
                    cons = get_info_v(row[7],cons_pattern)
                    if cons is not None:                        ins = cons                             #delly2 style
                    elif alt is not None and len(alt)>len(ref): ins = re.sub('[^actgnACTGN,]','',alt)  #clears manta alt style
                    if ins is not None:
                        new_svlen = len(ins); end = start+svlen
                        if new_svlen>svlen: svlen = new_svlen
                #genotype mining here...................................................
                if seq in seqs and svtype in sv_types and (filt=='PASS' or filt=='.') and svlen>=min_size and svlen<=max_size:
                    geno = 1 #parse and repair genotype information....
                    cnv  = get_info_v(row[7],cnv_pattern)
                    icn  = get_info_v(row[7],icn_pattern)
                    if len(row)>=10: #GT data is available
                        raw_geno = row[9].split(':')[0]
                        if raw_geno.find('|')>=0: geno = sum([(int(x) if x!='.' else 0) for x in raw_geno.split('|')])
                        else:                     geno = sum([(int(x) if x!='.' else 0) for x in raw_geno.split('/')])
                        if geno<1: geno = 1
                    elif cnv is not None: #cnvnator RD mining
                        cnv = float(cnv)
                        if cnv>0.0 and cnv<0.25:     geno = 2
                        elif cnv>=0.25 and cnv<1.0:  geno = 1
                        elif cnv>=1.0 and cnv<1.75:  geno = 1
                        elif cnv>=1.75:              geno = 2
                    elif icn is not None: #icn from GS mining
                        icn = int(icn)
                        if icn==0: geno = 2
                        elif icn==1: geno = 1
                    #parse and repair insertion sequence information for INS svtype
                    raw.add((seq,start,end,svtype,geno,ins,row[7]))
    # S = {}
    # for row in raw:
    #     if row[3] not in S: S[row[3]] = set([])
    #     S[row[3]].add(row)
    data = [list(row) for row in sorted(raw,key = lambda x: (x[0],x[1]))]
    D = {}
    for row in data:
        seq,svtype = row[0],row[3]
        if svtype in D:
            if seq in D[svtype]: D[svtype][seq] += [row[1:3]+row[4:]]
            else:                D[svtype][seq]  = [row[1:3]+row[4:]]
        else:                    D[svtype] = {seq:[row[1:3]+row[4:]]}
    for svtype in D:
        for seq in D[svtype]: D[svtype][seq] = sorted(D[svtype][seq],key=lambda x: x[0])
    return D

#for each cluster, get more data from ref sequence and bam files?
def bam_reader(bam_path):
    #GC content, MAPQ, micro-homolgy
    return True

def representative_cluster(c_idx,V,svtype,seq):
    idxs = {}
    for idx in c_idx:
        if idx[0] in idxs: idxs[idx[0]] += [idx[1]]
        else:              idxs[idx[0]]  = [idx[1]]
    for sid in idxs:
        if len(idxs[sid])>1:
            U = []
            for idx in idxs[sid]:
                U += [V[sid][svtype][seq][idx]]
            u = list(np.mean(U,axis=0,dtype=int))
            U = []
            for idx in idxs[sid]:
                xs = V[sid][svtype][seq][idx]
                U += [[abs(u[0]-xs[0])+abs(u[1]-xs[1])+abs(u[2]-xs[2]),idx]]
            idxs[sid] = sorted(U)[0][1]
        else: idxs[sid] = idxs[sid][0]
    return tuple(sorted([(sid,idxs[sid]) for sid in idxs]))

# def build_cluster_matrix(c_idx,V,svtype,seq,geno_i=2):
#     U,M,vs,cs = {},{},sorted(set(V).difference(set([0]))),[idx[0] for idx in c_idx]
#     for idx in c_idx: U[idx[0]] = V[idx[0]][svtype][seq][idx[1]][:(geno_i+1)]
#     u = np.mean(np.asarray([U[x] for x in U],dtype=np.float32),axis=0)
#     u[2] = np.median([U[x][2] for x in U])
#     u_span   = np.float32(abs(u[0]-u[1])+1)
#     u_center = np.float32(u[0]+(u_span//2+1))
#     for sid in vs:
#         if sid in cs:
#             s_span   = abs(U[sid][0]-U[sid][1])+1
#             s_center = (U[sid][0]+s_span//2+1)-u_center
#             M[sid] = [s_center,s_span,U[sid][2]]
#         else:         M[sid] = [np.nan,np.nan,0]
#     #add a redundant sid map for debuging purposes so we can track svtype trainig/building
#     return [np.int32(round(i)) for i in u],np.asarray([M[sid] for sid in sorted(M)],dtype=np.float32)

def build_true_matrix(X,M,T,t_ids):
    N,P = [],[]
    for i,j in t_ids:
        u = X[i]
        u_center = u[0]+abs(u[0]-u[1])//2+1
        t = T[j]
        t_span   = abs(t[0]-t[1])
        t_center = t[0]+t_span//2+1
        N += [M[i]]
        P += [[t_center-u_center,t_span,t[2]]]
    return np.asarray(N,dtype=np.int32),np.asarray(P,dtype=np.float32)

#compute the f1 score for each svtype between two vcams
#vcam = {sid_1:{svtype:seq:[[vc_1],[vc_2],...,[vc_x]]}}
def vcam_f1(V1,V2,over=0.5,geno_i=2,geno=False):
    svtypes,F1 = set(V1).union(set(V2)),{}
    for svtype in svtypes:
        if svtype in V1 and svtype in V2:
            a,b,x,y = 0,0,0,0
            for seq in V1[svtype]:
                A,B = set([]),set([])
                x += len(V1[svtype][seq])
                if seq in V2[svtype]:
                    y += len(V2[svtype][seq])
                    i_idx = get_intersection_idx(V1[svtype][seq],V2[svtype][seq],over=over)
                    for i,j in i_idx:
                        if geno and V1[svtype][seq][i][geno_i]==V2[svtype][seq][j][geno_i]: A.add(i); B.add(j)
                        else:                                                               A.add(i); B.add(j)
                    a,b = a+len(A),b+len(B)
            prec,rec,f1 = 0.0,0.0,0.0
            if x>0: prec = a/x
            if y>0: rec  = b/y
            if prec>0 or rec>0: f1 = 2*(prec*rec)/(prec+rec)
            F1[svtype] = f1
        else: F1[svtype] = 0.0
    return F1

def vcam_f1_matrix(VS,over,geno=False):
    M = {}
    for sample in VS:
        sids = sorted(VS[sample])
        for i in range(0,len(sids)-1,1):
            for j in range(i+1,len(sids),1):
                f1 = vcam_f1(VS[sample][sids[i]],VS[sample][sids[j]],over=over,geno=geno)
                k  = (sids[i],sids[j])
                for svtype in f1:
                    if svtype in M:
                        if k in M[svtype]: M[svtype][k] += [f1[svtype]]
                        else:              M[svtype][k]  = [f1[svtype]]
                    else:                  M[svtype]  = {k:[f1[svtype]]}
    for svtype in M:
        for k in M[svtype]:
            M[svtype][k] = np.mean(M[svtype][k])
    return M

def vcam_f1_sm_average(VS,v1,v2,svtypes=['DEL','DUP','INV','INS'],over=0.5):
    M,svtypes = {},set([])
    for sm in VS:
        if v1 in VS[sm] and v2 in VS[sm]:
            M[sm] = vcam_f1(VS[sm][v1],VS[sm][v2],over=over)
        elif (v1 in VS[sm] and v2 not in VS[sm]) or (v1 not in VS[sm] and v2 in VS[sm]): #missing values
            M[sm] = {svtype:0.0 for svtype in svtypes}                                   #are penalized
        svtypes.update(set(VS[sm][v1]))
    m = {svtype:[] for svtype in sorted(svtypes)}
    for sm in M:
        for svtype in M[sm]:
            m[svtype] += [M[sm][svtype]]
    for svtype in m:
        m[svtype] = np.mean(m[svtype])
    return M,m

def vcf_to_vcam(sample_path,out_dir,sids,seqs,sv_types,add_chr=True,verbose=False):
    if verbose: print('working on sample_path=%s'%sample_path)
    vcf_paths = {int(path.split('_S')[-1].split('.vcf')[0]):path for path in glob.glob(sample_path+'/*.vcf')}
    sample = sample_path.split('/')[-1]
    #-----------------------------------------------------------------------------------------
    if not os.path.exists(out_dir+'/vcam/%s.vcam.gz'%sample):
        if not os.path.exists(out_dir+'/vcam'): os.mkdir(out_dir+'/vcam')
        V = {sample:{}}
        for sid in vcf_paths:
            if sid in sids:
                if verbose: print('reading:%s'%vcf_paths[sid])
                V[sample][sid] = vcf_reader(vcf_paths[sid],seqs=seqs,sv_types=sv_types,add_chr=add_chr) #heavy lift here...
        with gzip.GzipFile(out_dir+'/vcam/%s.vcam.gz'%sample,'wb') as f: pickle.dump(V,f)
    else:
        with gzip.GzipFile(out_dir+'/vcam/%s.vcam.gz'%sample,'rb') as f: V = pickle.load(f)
    #----------------------------------------------------------------------------------------
    return V

def vcam_to_clust(V,out_dir,seqs,sv_sids,caller_pos=5,over=0.5,exclude_true=True,verbose=False):
    sample = sorted(V)[0]
    if not os.path.exists(out_dir+'/clust/%s.clust.gz'%sample):
        if not os.path.exists(out_dir+'/clust'): os.mkdir(out_dir+'/clust')
        C = {sample:{}}
        for svtype in sv_sids:
            if verbose: print('scaning svtype=%s'%svtype)
            for seq in seqs:
                X = []
                for sid in V[sample]:
                    if sid!=0:
                        if sid in sv_sids[svtype] and svtype in V[sample][sid] and seq in V[sample][sid][svtype]:
                            for i in range(len(V[sample][sid][svtype][seq])):
                                X += [V[sample][sid][svtype][seq][i]+[sid,i]]
                    elif not exclude_true:
                        if sid in sv_sids[svtype] and svtype in V[sample][sid] and seq in V[sample][sid][svtype]:
                            for i in range(len(V[sample][sid][svtype][seq])):
                                X += [V[sample][sid][svtype][seq][i]+[sid,i]]
                X = sorted(X,key=lambda x: x[0])
                if verbose: print('%s svs located on seq=%s'%(len(X),seq))
                cX = set([]) #this will be SV chords: cid=tuple(sorted((sid_1,idx_4),...,(sid_n,idx_m)))
                #(CYTHON) heavy lift here----------------------------------------------------------------------
                for i in range(len(X)): #need to use the recipricol overlap formula
                    span = int((1.0-over)*(X[i][1]-X[i][0]+1))
                    left,right = max(0,X[i][0]-span),X[i][1]+span
                    xs,ts = [],[tuple(X[i][caller_pos:])]
                    j = i-1
                    while j>0 and X[j][0]>left:
                        xs += [j]
                        j  -= 1
                    j = i+1
                    while j<len(X) and X[j][1]<right:
                        xs += [j]
                        j  += 1
                    for x in xs:
                        if overlap(X[i],X[x])>over:
                            ts += [tuple(X[x][caller_pos:])]
                    k = tuple(sorted(ts,key=lambda x: (x[0],x[1])))
                    cX.add(k)
                if len(cX)>1:
                    if svtype in C[sample]: C[sample][svtype][seq] = cX
                    else:                   C[sample][svtype] = {seq:cX}
            #(CYTHON) heavy lift here----------------------------------------------------------------------
        with gzip.GzipFile(out_dir+'/clust/%s.clust.gz'%sample,'wb') as f: pickle.dump(C,f)
    else:
        with gzip.GzipFile(out_dir+'/clust/%s.clust.gz'%sample,'rb') as f: C = pickle.load(f)
    #----------------------------------------------------------------------------------------
    return C

def vcam_clust_to_chord(V,C,out_dir):
    sample = sorted(V)[0]
    D = {sample:{}}
    if not os.path.exists(out_dir+'/chord/%s.chord.gz'%sample):
        if not os.path.exists(out_dir+'/chord'): os.mkdir(out_dir+'/chord')
        vs = get_sids_from_clust(C,sample)
        for svtype in C[sample]:
            D[sample][svtype] = {}
            for seq in C[sample][svtype]:
                M,X,L = [],[],[]
                for c_idx in C[sample][svtype][seq]:
                    x,m = utils.build_cluster_matrix(c_idx,V[sample],vs,svtype,seq)
                    X += [x]
                    M += [m]
                X = np.asarray(X,dtype=np.int32)   #chord coordinate/geno averages
                M = np.asarray(M,dtype=np.float32) #center (with respect to u center) and span matrix
                x_idx = np.argsort(X[:,0])         #coordinate sorting of clusters
                X = X[x_idx]
                M = M[x_idx]
                CL = list(C[sample][svtype][seq])
                for i in x_idx: L += [CL[i]]
                D[sample][svtype][seq] = {'chord':M,'v_idx':L,'u':X,'vs':vs[svtype]}
        with gzip.GzipFile(out_dir+'/chord/%s.chord.gz'%sample,'wb') as f: pickle.dump(D,f)
    else:
        with gzip.GzipFile(out_dir+'/chord/%s.chord.gz'%sample,'rb') as f: D = pickle.load(f)
    return D

def partition_true_chord(V,D,tid=0,t_over=0.5,geno_i=2,verbose=True):
    sample = sorted(D)[0]
    P = {sample:{}}
    if not os.path.exists(out_dir+'/train/%s.train.gz'%sample):
        if not os.path.exists(out_dir+'/train'): os.mkdir(out_dir+'/train')
        for svtype in D[sample]:
            n,m,a = 0,0,0
            P[sample][svtype] = {}
            for seq in D[sample][svtype]:
                P[sample][svtype][seq] = {}
                if tid in V[sample] and svtype in V[sample][tid] and seq in V[sample][tid][svtype]:
                    X     = D[sample][svtype][seq]['u']
                    M     = D[sample][svtype][seq]['chord']
                    L     = D[sample][svtype][seq]['v_idx']
                    T     = [v[:(geno_i+1)] for v in V[sample][tid][svtype][seq]]
                    t_ids = get_intersection_idx(X,T,over=t_over)
                    t_sdi = sorted(set(range(len(X))).difference(set([t[0] for t in t_ids])))
                    n += min(len(T),len(t_ids)) #need to unpack these...
                    m += len(X)
                    a += len(T)
                    if len(t_ids)>0:
                        P[sample][svtype][seq]['I'] = {'u':    X[[t[0]  for t in t_ids]],
                                                       't':    np.asarray([T[t[1]] for t in t_ids],dtype=np.int32),
                                                       'chord':M[[t[0]  for t in t_ids]],
                                                       'v_idx':[L[t[0]] for t in t_ids],
                                                       'vs':   D[sample][svtype][seq]['vs']}
                    P[sample][svtype][seq]['D1'] = {'u':     X[t_sdi],
                                                    'chord': M[t_sdi],
                                                    'v_idx': [L[t] for t in t_sdi],
                                                    'vs':    D[sample][svtype][seq]['vs']}
            if verbose:
                prec,rec,f1 = 0.0,0.0,0.0
                if a>0:             prec = n/a
                if m>0:             rec  = n/m
                if prec>0 or rec>0:   f1 = 2*(prec*rec)/(prec+rec)
                print('%s: prec=%s,rec=%s,f1=%s for sample=%s'%\
                              (svtype,round(prec,2),round(rec,2),round(f1,2),sample))
            with gzip.GzipFile(out_dir+'/train/%s.train.gz'%sample,'wb') as f: pickle.dump(P,f)
    else:
        with gzip.GzipFile(out_dir+'/train/%s.train.gz'%sample,'rb') as f: P = pickle.load(f)
    return P

def get_sids_from_clust(C,sample):
    vs = {}
    for svtype in C[sample]:
        vs[svtype] = set([])
        for seq in C[sample][svtype]:
            for c_idx in C[sample][svtype][seq]:
                for idx in c_idx:
                    vs[svtype].add(idx[0])
        vs[svtype] = sorted(vs[svtype].difference(set([0])))
    return vs

def get_sids_from_chord_map(DS,sids):
    vs = {}
    for sm in DS:
        for svtype in DS[sm]:
            if svtype not in vs: vs[svtype] = set([])
            for seq in DS[sm][svtype]:
                if 'I' in DS[sm][svtype][seq]:
                    for sid in DS[sm][svtype][seq]['I']['vs']:  vs[svtype].add(sid)
                if 'D1' in DS[sm][svtype][seq]:
                    for sid in DS[sm][svtype][seq]['D1']['vs']: vs[svtype].add(sid)
    for svtype in vs: vs[svtype] = [sids[x] for x in sorted(vs[svtype])]
    return vs

def get_nan_columns(X):
    xs = set([])
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            if np.isnan(X[:,i,j]).all(): xs.add(i); break
    return xs

def build_multi_imputer(DS,samples,seqs,svtype,n_jobs=4,ccp_alpha=0.01):
    print('center,span imputation for svtype=%s...'%svtype)
    M,T = [],[]
    for sample in DS:
        if sample in samples:
            if svtype in DS[sample]:
                for seq in DS[sample][svtype]:
                    if seq in seqs:
                        if 'I' in DS[sample][svtype][seq]:
                            M += [DS[sample][svtype][seq]['I']['chord']]
    if len(M)>0:
        M = np.concatenate(M)
        nan_idx = get_nan_columns(M)
        not_nan = sorted(set(range(M.shape[1])).difference(nan_idx))
        M = M[:,not_nan,:]
        M      = M.reshape(M.shape[0],M.shape[1]*M.shape[2])
        est = ExtraTreesRegressor(n_jobs=n_jobs,max_depth=8,min_samples_leaf=2,bootstrap=True,ccp_alpha=ccp_alpha)
        imp = IterativeImputer(estimator=est,max_iter=10,skip_complete=True,tol=0.01,random_state=r_seed)
        imp.fit(M)
    else: imp,not_nan = None,None
    return imp,not_nan

def chord_true(t,u):
    t_span   = t[1]-t[0]+1
    t_center = t[0]+t_span//2
    u_span   = u[1]-u[0]+1
    u_center = u[0]+u_span//2
    return [u_center-t_center,t_span,t[2]]

#X is the original chord mean,W is the imputed chord,
# P is the predicted center and span,Y is the true center and span
def center_span_diff(X,W,P,Y,geno=False):
    xy,wy,py = [],[],[]
    for i in range(len(Y)):
        x_span = X[i][1]-X[i][0]
        xy += [[abs(Y[i][0]-0),abs(Y[i][1]-x_span)]]
        w   = np.mean(W[i],axis=0,dtype=np.int32)
        wy += [[abs(Y[i][0]-w[0]),abs(Y[i][1]-w[1])]]
        py += [[abs(Y[i][0]-P[i][0]),abs(Y[i][1]-P[i][1])]]
    return xy,wy,py #mean(xy)>mean(wy)>mean(py) => jaccard(P,T) > jaccard(X,T)

def basic_score(y,p,w=1.0): #no genotyping here...
    a = 0
    for i in range(len(y)):
        if y[i]>0 and p[i]>0: a += 1
    prec,rec,f1,w1 = 0.0,0.0,0.0,0.0
    if sum(y)>0:   prec = a/sum([1 if x>0 else 0 for x in y])
    if sum(p)>0:   rec  = a/sum([1 if x>0 else 0 for x in p])
    if (prec+rec)>0:
        f1 = weighted_f1(prec, rec, 1.0)
        w1 = weighted_f1(prec,rec,w)
        #do jaccard_1D..................
    return prec,rec,f1,w1

def build_classifier_WY(WT,WD,Y,not_nan,nt_prop=4.0,scale_w=False,norm_w=True):
    t,d = WT.shape[0],WD.shape[0]
    if nt_prop is not None: n = min(int(round(t*nt_prop)),d)
    else:                   n = d
    ds_idx = np.random.choice(range(WD.shape[0]),n,replace=False)
    WD_sampled = WD[ds_idx]
    if scale_w:
        W_ = np.concatenate([WT,WD_sampled],dtype=np.float64)
        for i in range(W_.shape[2]):
            _min,_max = np.min(W_[:,:,i]),np.max(W_[:,:,i])
            _mag = _max-_min
            if _mag>0.0:    W_[:,:,i] = (W_[:,:,i]-_min)/(_mag)
            elif _max>0.0:  W_[:,:,i] = W_[:,:,i]/_max
    elif norm_w:
        W_ = np.concatenate([WT,WD_sampled],dtype=np.float64)
        for i in range(W_.shape[2]):
            mu,std = np.mean(W_[:,:,i]),np.std(W_[:,:,i])
            W_[:,:,i] -= mu
            if std>0.0: W_[:,:,i] /= std
    else: W_ = np.concatenate([WT,WD_sampled])
    W_ = W_.reshape(W_.shape[0],len(not_nan)*3)
    Y_ = np.concatenate([Y[:,2],np.zeros((WD_sampled.shape[0],),dtype=np.int32)])
    r_idx   = np.random.choice(range(Y_.shape[0]),Y_.shape[0],replace=False)
    return W_[r_idx],Y_[r_idx]

def train_classifier(DS,IM,svtype,train_samples,test_samples,train_seqs,test_seqs,props,iterations,r_seed):
    #training data --------------------------------------------------------------------------
    XT1,XD1,WT1,WD1,Y1 = partition_chord_data(DS,svtype,IM[svtype]['imp'],IM[svtype]['not_nan'],train_samples,train_seqs)
    #testing data ------------------------------------------------------------------------
    XT2,XD2,WT2,WD2,Y2 = partition_chord_data(DS,svtype,IM[svtype]['imp'],IM[svtype]['not_nan'],test_samples,test_seqs)
    vs = get_sids_from_chord_map(DS,sids)
    score = {svtype:{'model':None,'w1':0.0,'f1':0.0,'prec':0.0,'rec':0.0,'n':0,'m':0,'prop':0.0,'ensemble':vs[svtype]}}
    if not os.path.exists(out_dir+'/models/%s.model.pickle'%svtype):
        print('[%s training]: ensemble=(%s)'%(svtype,','.join(vs[svtype]))) #would like to know the sids used to impute and train....
        #training data --------------------------------------------------------------------------
        XT1,XD1,WT1,WD1,Y1 = partition_chord_data(DS,svtype,IM[svtype]['imp'],IM[svtype]['not_nan'],train_samples,train_seqs)
        #testing data ---------------------------------------------------------------------------
        XT2,XD2,WT2,WD2,Y2 = partition_chord_data(DS,svtype,IM[svtype]['imp'],IM[svtype]['not_nan'],test_samples,test_seqs)

        for prop in props: #1.0 is equal balance which will ensure prec is high...
            for i in range(iterations):
                if i%10==0: print('prop=%s,iteration=%s%s'%(prop,i,':'.join(['' for x in range(i//2)])))
                #check for sufficient distribution --> or use one-class modeling-----------------------------------------------------
                W_train,Y_train = build_classifier_WY(WT1,WD1,Y1,IM[svtype]['not_nan'],nt_prop=prop,scale_w=True)
                batches = [min(len(Y_train)//x[0]+1,x[1]) for x in [[2,128],[4,64],[8,32],[16,16],[32,8],[64,16]]]
                ada = AdaBoostClassifier(learning_rate=np.random.choice([1.0,0.9,0.8,0.7],1)[0],
                                         n_estimators=np.random.choice([50,100,150,200],1)[0],random_state=r_seed).fit(W_train,Y_train)
                his = HistGradientBoostingClassifier(max_iter=200,random_state=r_seed).fit(W_train,Y_train)
                layer_size_options = [(4,3),(5,3),(5,2),(6,2),(6,3),(7,3),(9,3),(16,3),(16,4),(32,4)]
                mlp = MLPClassifier(solver='adam', activation='relu',shuffle=True,
                                    alpha=np.random.choice([1.0E-6,2.5E-6,5.0E-6,1.0E-5],1)[0],
                                    batch_size=np.random.choice(batches,1)[0],
                                    hidden_layer_sizes=layer_size_options[np.random.choice(range(len(layer_size_options)),1)[0]],
                                    random_state=r_seed).fit(W_train,Y_train)
                models = [ada,his,mlp]
                #test the classifier::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                W_test,Y_test = build_classifier_WY(WT2,WD2,Y2,IM[svtype]['not_nan'],nt_prop=None,scale_w=True)
                for model in models:
                    P_test = model.predict(W_test)
                    prec,rec,f1,w1 = basic_score(Y_test,P_test,2.0)
                    model_str = model.__str__().split('(')[0]
                    if w1>score[svtype]['w1']:
                        print('::: w1 score: %s=%s , f1=%s, prec=%s, rec=%s:::'%\
                              (model_str,round(w1,2),round(f1,2),round(prec,2),round(rec,2)))
                        #function for this part: update_score[svtype...----------------------
                        score[svtype]['model'] = model
                        score[svtype]['prec']  = prec
                        score[svtype]['rec']   = rec
                        score[svtype]['f1']    = f1
                        score[svtype]['w1']    = w1
                        score[svtype]['n']     = sum(Y_test)
                        score[svtype]['m']     = sum(P_test)
                        score[svtype]['prop']  = prop
                        ensemble = vs[svtype]
                        score[svtype]['factors'] = tally_true_calls(W_test,model.predict(W_test),Y_test,ensemble)['T']
                        #calculate the contribution for this new model using its training WT1,Y1 yields...

        # #12/18/22 models are DEL@ 0.63 >0.54(old), DUP@ 0.27 >0.19(old), INV@ 0.76 >0.45(old)---------------------
        # #FusorSV notes: best DEL f1 is Lumpy @ 0.54, FusorSV INV f1 is @ 0.45, FusorSV DUP is @ 0.16 < (GS was 0.19)
        with open(out_dir+'/models/%s.model.pickle'%svtype,'wb') as f: pickle.dump(score[svtype],f)
    else:
        with open(out_dir+'/models/%s.model.pickle'%svtype,'rb') as f: score[svtype] = pickle.load(f)
    print('::::::::::final training results:::::::')
    print(score)
    return score

#takes a vector of labels = [0,1,2] and returns the index of the unique caller (otherwise -1)
def unique_caller(calls):
    one = False
    for i in range(len(calls)):
        if calls[i]>0:
            if one!=True: cid=i
            else:         cid=-1
            one = True
    return one

#give a caller index and clear/obfuscate its values across W
def clear_w_index(W,index,random=True):
    n = W.shape[1]//3
    WN = np.copy(W)
    WN = WN.reshape(WN.shape[0],n,3)
    WN[:,index,2] = 0.0 #zero out its decision...
    return WN.reshape(W.shape[0],W.shape[1])

def tally_true_calls(W,P,Y,ensemble):
    n = W.shape[1]//3
    T,M = {caller:0 for caller in ensemble},{caller:0 for caller in ensemble}
    T['total'],M['total'] = 0,0
    for i in range(len(P)):
        if P[i]>0:
            if Y[i]>0:
                calls = np.asarray(2*W[i].reshape((n,3)),dtype=int)[:,2]
                for j in range(len(calls)):
                    if calls[j]>0:
                        if ensemble[j] in T: T[ensemble[j]] += 1
                        else:                T[ensemble[j]]  = 1
                T['total'] += 1
        else:
            calls = np.asarray(2*W[i].reshape((n,3)),dtype=int)[:,2]
            for j in range(len(calls)):
                if calls[j]>0:
                    if ensemble[j] in M: M[ensemble[j]] += 1
                    else:                M[ensemble[j]]  = 1
            M['total'] += 1
    return {'T':T,'M':M}

#input  is DS = {sample:svtype:seq:{'I':{chord}, 'D':{chord}}
#output is transformed/imputed sub_sample/sub_seq data for classification
def partition_chord_data(DS,svtype,imp,not_nan,sub_samples,sub_seqs):
    XT,XD,WT,WD,Y = [],[],[],[],[]
    for sample in DS:
        if sample in sub_samples:
            for seq in DS[sample][svtype]:
                if seq in sub_seqs:
                    if 'I' in DS[sample][svtype][seq]:
                        x1 = DS[sample][svtype][seq]['I']['u']
                        yy = DS[sample][svtype][seq]['I']['t']
                        y1 = np.asarray([chord_true(yy[i],x1[i]) for i in range(len(yy))],dtype=np.int32)
                        #imputation::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                        m  = DS[sample][svtype][seq]['I']['chord']
                        m  = m[:,not_nan,:].reshape(m.shape[0],len(not_nan)*3)
                        if len(x1)>0:
                            w1 = np.int32(imp.transform(m).reshape(m.shape[0],len(not_nan),3))
                            XT += [x1]
                            WT += [w1]
                            Y  += [y1]
                    if 'D1' in DS[sample][svtype][seq]:
                        x1 = DS[sample][svtype][seq]['D1']['u']
                        m  = DS[sample][svtype][seq]['D1']['chord']
                        m  = m[:,not_nan,:].reshape(m.shape[0],len(not_nan)*3)
                        if len(x1)>0:
                            w1 = np.int32(imp.transform(m).reshape(m.shape[0],len(not_nan),3))
                            XD += [x1]
                            WD += [w1]
    if len(XT)>0: XT = np.concatenate(XT)
    if len(XD)>0: XD = np.concatenate(XD)
    if len(WT)>0: WT = np.concatenate(WT)
    if len(WD)>0: WD = np.concatenate(WD)
    if len(Y)>0:  Y  = np.concatenate(Y)
    return XT,XD,WT,WD,Y


def preprocess_predict_data(DS,svtype,imp,not_nan,sub_samples,sub_seqs,scale_w=False,norm_w=False,hybrid_w=True):
    XTD,WTD = [],[]
    for sample in DS:
        if sample in sub_samples:
            for seq in DS[sample][svtype]:
                if seq in sub_seqs:
                    x1 = DS[sample][svtype][seq]['u']
                    #imputation:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                    m  = DS[sample][svtype][seq]['chord']
                    m  = m[:,not_nan,:].reshape(m.shape[0],len(not_nan)*3)
                    if len(x1)>0:
                        w1 = np.int32(imp.transform(m).reshape(m.shape[0],len(not_nan),3))
                        XTD += [x1]
                        WTD += [w1]
    if len(XTD)>0: XTD = np.concatenate(XTD)
    if len(WTD)>0: WTD = np.concatenate(WTD)
    #-------------------------------------------------------------------------------------
    n = WTD.shape[0]
    if scale_w:
        W_ = np.asarray(WTD,dtype=np.float64)
        for i in range(W_.shape[2]):
            _min,_max = np.min(W_[:,:,i]),np.max(W_[:,:,i])
            _mag = _max-_min
            if _mag>0.0:    W_[:,:,i] = (W_[:,:,i]-_min)/(_mag)
            elif _max>0.0:  W_[:,:,i] = W_[:,:,i]/_max
    if norm_w:
        W_ = np.asarray(WTD,dtype=np.float64)
        for i in range(W_.shape[2]):
            mu,std = np.mean(W_[:,:,i]),np.std(W_[:,:,i])
            W_[:,:,i] -= mu
            if std>0.0: W_[:,:,i] /= std
    if hybrid_w:
        W_ = np.asarray(WTD, dtype=np.float64)
        mu,std = np.mean(W_[:,:,0]),np.std(W_[:,:,0])
        W_[:,:,0] -= mu
        if std>0.0: W_[:,:,0] /= std
        _mag = int(1e6)-50
        W_[:,:,1] = (W_[:,:,1]-50)/_mag
        W_[:,:,2] /= 2.0
    else: W_ = WTD
    W_ = W_.reshape(W_.shape[0],len(not_nan)*3)
    return W_

def print_call_hist(C,min_len=50,max_len=int(1e6)):
    H = {}
    for sm in C:
        H[sm] = {}
        for svtype in C[sm]:
            for row in C[sm][svtype]:
                if min_len<=abs(row[2]-row[1])<=max_len:
                    if svtype not in H[sm]: H[sm][svtype] = []
                    H[sm][svtype] += [row]
        DEL = (len(H[sm]['DEL']) if 'DEL' in H[sm] else 0)
        INV = (len(H[sm]['INV']) if 'INV' in H[sm] else 0)
        INS = (len(H[sm]['INS']) if 'INS' in H[sm] else 0)
        DUP = (len(H[sm]['DUP']) if 'DUP' in H[sm] else 0)
        print('sm=%s: DEL=%s, INS=%s, INV=%s, DUP=%s, total=%s'%(sm, DEL,INS,INV,DUP,DEL+INS+INV+DUP))

def pred_clust(V,sv_sids,caller_pos=5,c_over=0.5):
    C,X = {},{}
    for sm in V:
        for svtype in V[sm]:
            if svtype not in X: X[svtype] = {}
            for row in V[sm][svtype]:
                if row[0] not in X[svtype]: X[svtype][row[0]] = []
                X[svtype][row[0]] += [row[1:-1]+[set([(v[0],'%s_%s'%(v[1],sm)) for v in row[-1]])]]
    for svtype in X:
        for seq in X[svtype]: X[svtype][seq] = sorted(X[svtype][seq],key=lambda x: x[0])
    for svtype in X:
        for seq in X[svtype]:
            Y = X[svtype][seq]
            cY = set([]) #this will be SV chords: cid=tuple(sorted((sid_1,idx_4),...,(sid_n,idx_m)))
            for i in range(len(Y)): #need to use the recipricol overlap formula
                span = int((1.0-c_over)*(Y[i][1]-Y[i][0]+1))
                left,right = max(0,Y[i][0]-span),Y[i][1]+span
                xs,ts = [],set([i])
                j = i-1
                while j>0 and Y[j][0]>left:
                    xs += [j]
                    j  -= 1
                j = i+1
                while j<len(Y) and Y[j][1]<right:
                    xs += [j]
                    j  += 1
                for x in xs:
                    if overlap(Y[i],Y[x])>c_over:
                        ts.add(x)
                k = tuple(sorted(ts))
                cY.add(k)
            if len(cY)>1:
                if svtype in C: C[svtype][seq] = cY
                else:           C[svtype] = {seq:cY}
    return C,X

def write_vcf_files():
    return True

if __name__ == '__main__':
    des = """
    Prototype Structural Variation Ensemble Analysis (VCF4.0+)
    Timothy James Becker, 12/15/2022-08/22/24, version=0.0.1
    training requirements: ~32GB RAM and 4 cores minimum"""
    parser = argparse.ArgumentParser(description=des,formatter_class=argparse.RawTextHelpFormatter)
    in_dir_des = """study folder comprised of folders (one folder for each sample id) each of which has multple uniform caller id tagged vcf files:
    [EX] /path/sampleW/sampleW_S4.vcf implies that there is a VCF file for caller id (sid)=4 for sampleW\t[None]\n
    (Note) To use on sample1,sample2,sample3 with caller id (sid=4), you would have the following folders and files set up:
    /path/sample1/sample1_S4.vcf
    /path/sample2/sample2_S4.vcf
    /path/sample3/sample3_S4.vcf
    [Note] can be a comma-seperated set of folders to use multiple studies...
    """
    parser.add_argument('--in_dirs',type=str, help=in_dir_des)
    parser.add_argument('--out_dir',type=str, help='output directory to save vcam,cluster,model,vcf/ into\t[None]')
    parser.add_argument('--pred_dirs',type=str, help='predict directory to call new VCFs from using models\t[None]')
    parser.add_argument('--seqs',type=str,help='comma seperated chrom listing\t[taken from VCF files..]')
    parser.add_argument('--sv_mask',type=str,help='BED3/6 format of excluded regions (pre-merge)\t[None]')
    parser.add_argument('--sv_types',type=str,help='comma seperated sv types\t[DEL,DUP,INV,INS]')
    stage_map_des = """JSON map of caller ids to stage names (and back):
    stage_map_json_file -> {0:'True',4:'BreakDancer',8:'CNVpytor',9:'ERDS',10:'CNVnator',11:'Delly2',13:'GATK',
                            14:'GenomeSTRiP',17:'Hydra',18:'LUMPY',35:'BreakSeq',36:'Pindel',38:'Tigra',
                            48: 'Manta',50:'TensorSV',54:'GRIDSS2',55:'MELT'}\t[None]
    """
    parser.add_argument('--stage_map',type=str,help=stage_map_des)
    parser.add_argument('--caller_include_list',type=str,help='comma seperated caller id list to include from test/training\t[4,10,18,48,54,55]')
    parser.add_argument('--c_over',type=float,help='reciprocal overlap used to cluster\t[0.7]')
    parser.add_argument('--t_over',type=float,help='reciprocal overlap used for scoring metrics to true data\t[0.5]')
    parser.add_argument('--split',type=float,help='train/test proportion\t[0.3]')
    parser.add_argument('--iterations',type=int,help='number of total iterations to run\t[100]')
    parser.add_argument('--f1_w',type=float,help='training f1 weight between prec and recall\t[1.0]')
    parser.add_argument('--cpus',type=int,help='number of cpus for processing\t[1]')
    train_class_prop_des = """comma seperated list of training class proportions\t[1.0]
    (Note: proportions are negative label to positive label: GT 0/0 versus 0/1 or 1/1)
    """
    parser.add_argument('--training_class_prop',type=str,help=train_class_prop_des)
    parser.add_argument('--rand_seed',type=int,help='random_seed\t[100]')
    parser.add_argument('--flag',action='store_true',help='placeholder flag option\t[False]')
    args = parser.parse_args()

    #G1KP3 sample VCF folder from FusorSV data set----------------------
    if args.in_dirs is not None: in_dirs = [path+'/*' for path in args.in_dirs.split(',')]
    else: print('input directory was not given...'); raise IOError
    if args.out_dir is not None: out_dir = args.out_dir
    else: print('output directory was not given...'); raise IOError

    #-------------------------------------------------------------------------
    if args.pred_dirs is not None: pred_dirs = [path + '/*' for path in args.pred_dirs.split(',')]
    else: print('prediction directory was not given, building models only...'); pred_dir = None
    #-------------------------------------------------------------------------

    if not os.path.exists(out_dir): os.mkdir(out_dir)
    if args.seqs is not None:     seqs = args.seqs.split(',')
    else:                         seqs = None #need to get from the inputs...
    if args.sv_mask is not None:  sv_mask = args.sv_mask
    else:                         sv_mask = {}
    if args.sv_types is not None: sv_types = args.sv_types.split(',')
    else:                         sv_types = ['DEL','DUP','INV','INS']
    if args.stage_map is not None:
        with open(args.stage_map,'r') as f:
            raw = json.load(f)
            sids = {int(k):raw[k] for k in raw}
    else:
        sids = {0:'True',-1:'FusorSV',-2:'reSolVer',4:'BreakDancer',8:'CNVpytor',9:'ERDS',10:'CNVnator',11:'Delly2',14:'GenomeSTRiP',
                17:'Hydra',18:'LUMPY',35:'Breakseq2',36:'Pindel',38:'TigraSV', 48:'Manta',50:'TensorSV',54:'GRIDSS',55:'MELT',
                60:'Sniffles2',61:'PBSV',62:'NanoVar',63:'PBHoney'}
        SVSIDS = {'DEL':[4,9,10,11,18],'DUP':[4,8,9,10,14,17,18,48],'INV':[4,11,17,18,48,54],'INS':[11,48,54,55]}
        sv_sids = {}
        for sv in SVSIDS:
            sv_sids[sv] = {}
            for sid in SVSIDS[sv]:
                sv_sids[sv][sid] = sids[sid]
    if args.caller_include_list is not None: #redo this so you can have per svtype include_list
        call_ids = [-1]+[int(x) for x in args.caller_include_list.split(',')]
        _sids    = {}
        for cid in call_ids:
            if cid in sids: _sids[cid] = sids[cid]
        if 0 not in _sids:  _sids[0]   = sids[0]
        sids = _sids
    samples = []
    for in_dir in in_dirs: samples += glob.glob(in_dir)
    samples = sorted(samples)
    if args.c_over is not None:     c_over      = args.c_over
    else:                           c_over      = 0.5
    if args.t_over is not None:     t_over      = args.t_over
    else:                           t_over      = 0.75
    if args.split is not None:      split       = args.split
    else:                           split       = 0.70
    if args.iterations is not None: iterations  = args.iterations
    else:                           iterations  = 30
    if args.cpus is not None:       cpus        = args.cpus
    else:                           cpus        = 12
    if args.f1_w is not None:       f1_w        = args.f1_w
    else:                           f1_w        = 1.0
    if args.training_class_prop is not None:
        training_class_prop = [float(x) for x in args.training_class_prop.split(',')]
    else:
        training_class_prop = [None]
    if args.rand_seed is not None:  r_seed = args.rand_seed
    else:                           r_seed      = 100

    np.random.seed(r_seed)
    SS = partition_samples_seqs(samples,seqs,split=split)
    train_samples,test_samples,valid_samples = SS['samples']
    train_seqs,test_seqs,valid_seqs = SS['seqs']
    samples = sorted(train_samples+test_samples+valid_samples)
    train_samples = [sm.split('/')[-1] for sm in train_samples]
    test_samples  = [sm.split('/')[-1] for sm in test_samples]
    valid_samples = [sm.split('/')[-1] for sm in valid_samples]

    partitions = [[] for i in range(cpus)]
    for i in range(len(samples)): partitions[i%cpus] += [samples[i]]
    DS,CS,VS = utils.process_prepare_training_samples(out_dir,partitions,seqs,sv_types,sids,sv_sids) #test out ||

    IM = {}
    print('starting chord imputation via center/span regression...')
    #----------------------------------------------------------------------------------------------------------
    for svtype in sorted(sv_types):
        if not os.path.exists(out_dir+'/models'): os.mkdir(out_dir+'/models')
        if not os.path.exists(out_dir+'/models/%s.imp.pickle'%svtype):
            print('did not find [%s.imp.pickle] model, imputing...'%svtype)
            imp,not_nan = build_multi_imputer(DS,train_samples,train_seqs,svtype,n_jobs=cpus) #make one pass at imputation...
            IM[svtype] = {'imp':imp,'not_nan':not_nan}
            with open(out_dir+'/models/%s.imp.pickle'%svtype,'wb') as f:
                pickle.dump({'imp':IM[svtype]['imp'],'not_nan':IM[svtype]['not_nan']},f)
        else:
            print('found [%s.imp.pickle] model, loading...'%svtype)
            with open(out_dir+'/models/%s.imp.pickle'%svtype,'rb') as f:
                IM[svtype] = pickle.load(f) #IM[svtype]['imp'], IM[svtype]['not_nan']

    start = time.time()
    prop = training_class_prop[0]
    score = utils.process_train_classifier(out_dir,DS,IM,sids,sv_types,train_samples,test_samples,
                                           train_seqs,test_seqs,training_class_prop,iterations,f1_w,r_seed,cpus)
    stop = time.time()
    print('training in %s min(s)'%((stop-start)/60))

    #see if there is a prediction folder of VCFs now, we can then load up the data and run predictions on clusters
    #from those predictions we can produce the VCF file by reversing the clusters...

    #now we write a utils.process_predict_classifier()
    def predict_classifier():
        P_pred,V_pred = {},{}
        return V_pred

    if pred_dirs is not None:
        pred_samples = []
        for pred_dir in pred_dirs: pred_samples += glob.glob(pred_dir)
        ps = []
        for pred in pred_samples:
            if pred.split('/')[-1] in valid_samples: ps += [pred]
            #ps += [pred]
        pred_samples = sorted(ps)
        DS,CS,VS = utils.worker_prepare_prediction_samples(out_dir,pred_samples,seqs,sv_types,sids,sv_sids,c_over)
        IM = {}
        for svtype in sv_types:
            with open(out_dir+'/models/%s.imp.pickle'%svtype,'rb') as f:
                IM[svtype] = pickle.load(f)

        #work through each sample, svtype, and seq to retireve the information to make a VCF
        P_pred,V_pred = {},{}
        for p_sample in DS:
            P_pred[p_sample],V_pred[p_sample] = {},{}
            print('prediction on sample=%s'%p_sample)
            for svtype in DS[p_sample]:
                imp,not_nan = IM[svtype]['imp'],IM[svtype]['not_nan']
                P_pred[p_sample][svtype],V_pred[p_sample][svtype] = {},[]
                for seq in DS[p_sample][svtype]:
                    ds_h   = DS[p_sample][svtype][seq]
                    vs     = DS[p_sample][svtype][seq]['vs']
                    W_pred = preprocess_predict_data(DS,svtype,imp,not_nan,[p_sample],[seq],hybrid_w=True)
                    P_pred[p_sample][svtype][seq] = score[svtype]['model'].predict(W_pred)
                    for i in range(len(P_pred[p_sample][svtype][seq])): #select positive predictions
                        vc = P_pred[p_sample][svtype][seq][i]
                        if vc>0:
                            mu = ds_h['u'][i]
                            cs = [[v[0],v[1]]+VS[p_sample][v[0]][svtype][seq][v[1]] for v in ds_h['v_idx'][i]]
                            method = set([(c[0],c[1]) for c in cs])
                            if svtype=='INS':
                                ins_seq = []
                                for c in cs:
                                    if c[0]==48 or c[0]==54: ins_seq += [c[5]]
                                    if c[0]==55: ins_seq = [c[6].split('MEINFO=')[-1].split(',')[0].split(';')[0]]; break
                                V_pred[p_sample][svtype] += [[seq,mu[0],mu[1],svtype,mu[2],ins_seq[0],method]]
                            if svtype=='INV':
                                sr,pe = [],[]
                                for c in cs:
                                    if c[0]==11 or c[0]==18:
                                        rs  = c[6].split(';SR=')[-1].split(';')[0]
                                        sr += [(int(rs) if rs.isdigit() else 0)]
                                        ep  = c[6].split(';PE=')[-1].split(';')[0]
                                        pe += [(int(ep) if ep.isdigit() else 0)]
                                if len(sr)>0: sr = int(round(np.mean(sr)))
                                else:         sr = 0
                                if len(pe)>0: pe = int(round(np.mean(pe)))
                                else:         pe = 0
                                V_pred[p_sample][svtype] += [[seq,mu[0],mu[1],svtype,mu[2],'SR=%s;PE=%s'%(sr,pe),method]]
                            if svtype=='DEL':
                                sr,pe,rd = [],[],[]
                                for c in cs:
                                    if c[0]==11 or c[0]==18:
                                        rs  = c[6].split(';SR=')[-1].split(';')[0]
                                        sr += [(int(rs) if rs.isdigit() else 0)]
                                        ep  = c[6].split(';PE=')[-1].split(';')[0]
                                        pe += [(int(ep) if ep.isdigit() else 0)]
                                    if c[0]==8 or c[0]==10:
                                        rd += [float(c[6].split('RD=')[-1].split(';')[0])]
                                if len(sr)>0: sr = int(round(np.mean(sr)))
                                else:         sr = 0
                                if len(pe)>0: pe = int(round(np.mean(pe)))
                                else:         pe = 0
                                if len(rd)>0: rd = int(round(np.mean(rd)))
                                else: rd = 2-(mu[2])
                                V_pred[p_sample][svtype] += [[seq,mu[0],mu[1],svtype,mu[2],'SR=%s;PE=%s;RD=%s'%(sr,pe,rd),method]]
                            if svtype=='DUP':
                                sub,rd,sr,pe = None,[],[],[]
                                for c in cs:
                                    if c[0]==11 or c[0]==18:
                                        rd += [2]
                                        if sub is None: sub = 'TANDEM'
                                        rs  = c[6].split(';SR=')[-1].split(';')[0]
                                        sr += [(int(rs) if rs.isdigit() else 0)]
                                        ep  = c[6].split(';PE=')[-1].split(';')[0]
                                        pe += [(int(ep) if ep.isdigit() else 0)]
                                    if c[0]==8 or c[0]==10:
                                        rd += [float(c[6].split('RD=')[-1].split(';')[0])]
                                        sub = 'DISPERSED'
                                if len(sr)>0: sr = int(round(np.mean(sr)))
                                else:         sr = 0
                                if len(pe)>0: pe = int(round(np.mean(pe)))
                                else:         pe = 0
                                if len(rd)>0: rd = int(round(np.mean(rd)))
                                else: rd = 2
                                V_pred[p_sample][svtype] += [[seq,mu[0],mu[1],svtype,mu[2],'%s;RD=%s'%(sub,rd),method]]
        print_call_hist(V_pred)

        # #make a vcam for V_pred and call vcam_f1(VS[sm][0],VS[sm][-2],over=0.5)
        sids[-2] = "reSolVer"
        for sm in V_pred:
            for svtype in V_pred[sm]:
                for row in V_pred[sm][svtype]:
                    seq = row[0]
                    if -2 not in VS[sm]:              VS[sm][-2] = {}
                    if svtype not in VS[sm][-2]:      VS[sm][-2][svtype] = {}
                    if seq not in VS[sm][-2][svtype]: VS[sm][-2][svtype][seq] = []
                    VS[sm][-2][svtype][seq] += [[row[1],row[2],row[3],row[5],row[6]]]
        for v in vs+[-1,-2]:
            print('%s'%sids[v],vcam_f1_sm_average(VS,0,v)[1])















