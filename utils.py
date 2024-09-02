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
simplefilter("ignore", category=ConvergenceWarning)
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt

result_list = []
def collect_results(result):
    return result_list.append(result)

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
        print(vs)
        for svtype in C[sample]:
            D[sample][svtype] = {}
            for seq in C[sample][svtype]:
                M,X,L = [],[],[]
                for c_idx in C[sample][svtype][seq]:
                    x,m = build_cluster_matrix(c_idx,V[sample],vs,svtype,seq)
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

def build_cluster_matrix(c_idx,V,vs,svtype,seq,geno_i=2):
    U,M,cs = {},{},[idx[0] for idx in c_idx]
    for idx in c_idx: U[idx[0]] = V[idx[0]][svtype][seq][idx[1]][:(geno_i+1)]
    u = np.mean(np.asarray([U[x] for x in U],dtype=np.float32),axis=0)
    u[2] = np.median([U[x][2] for x in U])
    u_span   = np.float32(abs(u[0]-u[1])+1)
    u_center = np.float32(u[0]+(u_span//2+1))
    for sid in vs[svtype]:
        if sid in cs:
            s_span   = abs(U[sid][0]-U[sid][1])+1
            s_center = (U[sid][0]+s_span//2+1)-u_center
            M[sid] = [s_center,s_span,U[sid][2]]
        else:         M[sid] = [np.nan,np.nan,0]
    return [np.int32(round(i)) for i in u],np.asarray([M[sid] for sid in sorted(M)],dtype=np.float32)

def partition_true_chord(V,D,out_dir,tid=0,t_over=0.5,geno_i=2,verbose=True):
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

def partition_chord_data(DS,svtype,imp,not_nan,sub_samples,sub_seqs):
    XT,XD,WT,WD,Y = [],[],[],[],[]
    for sample in DS:
        if sample in sub_samples:
            if svtype in DS[sample]:
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

def get_min_max_vs(DS,svtype,limit=int(1e6)):
    min_vs,max_vs = [0 for i in range(limit)],[]
    for sample in DS:
        if svtype in DS[sample]:
            for seq in DS[sample][svtype]:
                if 'I' in DS[sample][svtype][seq]:
                    vs = DS[sample][svtype][seq]['I']['vs']
                    if len(vs)>len(max_vs): max_vs = vs
                    if len(vs)<len(min_vs): min_vs = vs
                if 'D1' in DS[sample][svtype][seq]:
                    vs = DS[sample][svtype][seq]['D1']['vs']
                    if len(vs)>len(max_vs): max_vs = vs
                    if len(vs)<len(min_vs): min_vs = vs
                if 'vs' in DS[sample][svtype][seq]:
                    vs = DS[sample][svtype][seq]['vs']
                    if len(vs)>len(max_vs): max_vs = vs
                    if len(vs)<len(min_vs): min_vs = vs
    return min_vs,max_vs

def chord_true(t,u):
    t_span   = t[1]-t[0]+1
    t_center = t[0]+t_span//2
    u_span   = u[1]-u[0]+1
    u_center = u[0]+u_span//2
    return [u_center-t_center,t_span,t[2]]

def weighted_f1(prec,rec,p_w=1.0):
    return (p_w+1)/(p_w/prec+1/rec)

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

def build_classifier_WY(WT,WD,Y,not_nan,nt_prop=4.0,scale_w=False,norm_w=False,hybrid_w=False):
    t,d = WT.shape[0],WD.shape[0]
    if nt_prop is not None: n = min(int(round(t*nt_prop)),d)
    else:                   n = d
    ds_idx = np.random.choice(range(WD.shape[0]),n,replace=False)
    WD_sampled = WD[ds_idx]
    if scale_w:
        W_ = np.concatenate([WT,WD_sampled],dtype=np.float64)
        #0 is the center point
        for i in range(W_.shape[2]):
            _min,_max = np.min(W_[:,:,i]),np.max(W_[:,:,i])
            _mag = _max-_min
            if _mag>0.0:    W_[:,:,i] = (W_[:,:,i]-_min)/(_mag)
            elif _max>0.0:  W_[:,:,i] = W_[:,:,i]/_max
    if norm_w:
        W_ = np.concatenate([WT,WD_sampled],dtype=np.float64)
        for i in range(W_.shape[2]):
            mu,std = np.mean(W_[:,:,i]),np.std(W_[:,:,i])
            W_[:,:,i] -= mu
            if std>0.0: W_[:,:,i] /= std
    if hybrid_w:
        W_ = np.concatenate([WT, WD_sampled], dtype=np.float64)
        mu,std = np.mean(W_[:,:,0]),np.std(W_[:,:,0])
        W_[:,:,0] -= mu
        if std>0.0: W_[:,:,0] /= std
        _mag = int(1e6)-50
        W_[:,:,1] = (W_[:,:,1]-50)/_mag
        W_[:,:,2] /= 2.0
    else: W_ = np.concatenate([WT,WD_sampled])
    W_ = W_.reshape(W_.shape[0],len(not_nan)*3)
    Y_ = np.concatenate([Y[:,2],np.zeros((WD_sampled.shape[0],),dtype=np.int32)])
    r_idx   = np.random.choice(range(Y_.shape[0]),Y_.shape[0],replace=False)
    return W_[r_idx],Y_[r_idx]

def worker_train_classifier(out_dir,DS,IM,sids,svtype,train_samples,test_samples,train_seqs,test_seqs,prop,iterations,f1_w,r_seed):
    #out_dir,DS,IM,sids,svtype,train_samples,test_samples,train_seqs,test_seqs,None,iterations,f1_w,r_seed
    vs = get_sids_from_chord_map(DS,sids)
    score = {svtype:{'model':None,'w1':0.0,'f1':0.0,'prec':0.0,'rec':0.0,'n':0,'m':0,'prop':0.0,'ensemble':vs[svtype]}}
    if not os.path.exists(out_dir+'/models/%s.%s.model.pickle'%(svtype,prop)):
        #training data --------------------------------------------------------------------------
        XT1,XD1,WT1,WD1,Y1 = partition_chord_data(DS,svtype,IM[svtype]['imp'],IM[svtype]['not_nan'],train_samples,train_seqs)
        #testing data ---------------------------------------------------------------------------
        XT2,XD2,WT2,WD2,Y2 = partition_chord_data(DS,svtype,IM[svtype]['imp'],IM[svtype]['not_nan'],test_samples,test_seqs)
        print('[%s.%s training]: ensemble=(%s)'%(svtype,prop,','.join(vs[svtype]))) #would like to know the sids used to impute and train....
        for i in range(iterations):
            #check for sufficient distribution --> or use one-class modeling-----------------------------------------------------
            W_train,Y_train = build_classifier_WY(WT1,WD1,Y1,IM[svtype]['not_nan'],nt_prop=prop,hybrid_w=True)
            if W_train.shape[0]%2>0:
                W_train = W_train[:-1]
                Y_train = Y_train[:-1]
            batches = [min(len(Y_train)//x[0]+1,x[1]) for x in [[2,128],[4,64],[8,32],[16,16],[32,8],[64,16]]]
            models = []
            ada = AdaBoostClassifier(learning_rate=np.random.choice([1.0,0.9,0.8,0.7],1)[0],
                                     n_estimators=np.random.choice([50,100,150,200],1)[0],random_state=r_seed).fit(W_train,Y_train)
            models += [ada]
            try:
                his = HistGradientBoostingClassifier(max_iter=100,random_state=r_seed).fit(W_train,Y_train)
                models += [his]
            except Exception as e: print(e) #weird histboosting split issues...
            layer_size_options = [(4,3),(5,3),(5,2),(6,2),(6,3),(7,3),(9,3),(16,3),(16,4),(32,4)]
            mlp = MLPClassifier(solver='adam', activation='relu',shuffle=True,
                                alpha=np.random.choice([1.0E-6,2.5E-6,5.0E-6,1.0E-5],1)[0],
                                batch_size=np.random.choice(batches,1)[0],
                                hidden_layer_sizes=layer_size_options[np.random.choice(range(len(layer_size_options)),1)[0]],
                                random_state=r_seed).fit(W_train,Y_train)
            models += [mlp]
            #test the classifier::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
            W_test,Y_test = build_classifier_WY(WT2,WD2,Y2,IM[svtype]['not_nan'],nt_prop=None,hybrid_w=True)
            for model in models:
                #if i % 10 == 0: print('predicting sv=%s prop=%s,iteration=%s%s'%(svtype,prop,i,':'.join(['' for x in range(i//2)])))
                P_test = model.predict(W_test)
                prec,rec,f1,w1 = basic_score(Y_test,P_test,f1_w)
                model_str = model.__str__().split('(')[0]
                if w1>score[svtype]['w1']:
                    print('::: %s.%s w1 score: %s=%s , f1=%s, prec=%s, rec=%s:::'%\
                          (svtype, prop, model_str,round(w1,2),round(f1,2),round(prec,2),round(rec,2)))
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
        with open(out_dir+'/models/%s.%s.model.pickle'%(svtype,prop),'wb') as f: pickle.dump(score[svtype],f)
    else:
        with open(out_dir+'/models/%s.%s.model.pickle'%(svtype,prop),'rb') as f: score[svtype] = pickle.load(f)
    return score

def process_train_classifier(out_dir,DS,IM,sids,sv_types,train_samples,test_samples,
                             train_seqs,test_seqs,props,iterations,f1_w=1.0,r_seed=100,cpus=12):
    global result_list
    p2 = mp.Pool(processes=cpus)
    for svtype in sv_types:
        for prop in props:
            print('sv=%s,prop=%s'%(svtype,prop))
            p2.apply_async(worker_train_classifier,
                           args=(out_dir,DS,IM,sids,svtype,train_samples,test_samples,train_seqs,test_seqs,prop,iterations,f1_w,r_seed),
                           callback=collect_results)
    p2.close()
    p2.join()

    score = {}
    for result in result_list:
        for svtype in result:
            if svtype not in score: score[svtype] = result[svtype]
            elif score[svtype]['w1']<result[svtype]['w1']: score[svtype] = result[svtype]
    result_list = []
    for svtype in score:
        print('::::::::::final training results:::::::')
        print(score[svtype])
    return score

def worker_prepare_training_samples(out_dir,samples,seqs,sv_types,sids,cluster_sids,cluster_over,true_over):
    DS,CS,VS = {},{},{}
    for sample_path in sorted(samples): #can do this in ||
        sample = sample_path.split('/')[-1]
        VS[sample] = vcf_to_vcam(sample_path,out_dir,sids,seqs,sv_types,verbose=True)
        CS[sample] = vcam_to_clust(VS[sample],out_dir,seqs,cluster_sids,over=cluster_over,verbose=True)
        DS[sample] = vcam_clust_to_chord(VS[sample],CS[sample],out_dir)
        DS[sample] = partition_true_chord(VS[sample],DS[sample],out_dir,t_over=true_over)[sample]
        CS[sample] = CS[sample][sample]
        VS[sample] = VS[sample][sample]
    for svtype in sv_types: #check to ensure all callers are harmonized
        min_vs,max_vs = get_min_max_vs(DS,svtype)
        if min_vs!=max_vs: #expansion is needed-----------------------------------
            max_vs_idx = {max_vs[i]: i for i in range(len(max_vs))} #expand to max
            for sm in DS:
                if svtype in DS[sm]:
                    for seq in DS[sm][svtype]:
                        if 'I' in DS[sm][svtype][seq]:
                            vs = DS[sm][svtype][seq]['I']['vs']
                            vs_idx = {vs[i]:i for i in range(len(vs))}
                            idx_vs = {i:vs[i] for i in range(len(vs))}
                            if len(vs)<len(max_vs):
                                chord = DS[sm][svtype][seq]['I']['chord']
                                clean = np.zeros((chord.shape[0],len(max_vs),chord.shape[2]),dtype=chord.dtype)+np.nan
                                for i in range(len(chord)):
                                    for j in idx_vs:
                                        clean[i][max_vs_idx[idx_vs[j]]] = chord[i][j]
                                DS[sm][svtype][seq]['I']['chord'] = clean
                                DS[sm][svtype][seq]['I']['vs']    = max_vs
                        if 'D1' in DS[sm][svtype][seq]:
                            vs = DS[sm][svtype][seq]['D1']['vs']
                            vs_idx = {vs[i]:i for i in range(len(vs))}
                            idx_vs = {i:vs[i] for i in range(len(vs))}
                            if len(vs)<len(max_vs):
                                chord = DS[sm][svtype][seq]['D1']['chord']
                                clean = np.zeros((chord.shape[0],len(max_vs),chord.shape[2]),dtype=chord.dtype)+np.nan
                                for i in range(len(chord)):
                                    for j in idx_vs:
                                        clean[i][max_vs_idx[idx_vs[j]]] = chord[i][j]
                                DS[sm][svtype][seq]['D1']['chord'] = clean
                                DS[sm][svtype][seq]['D1']['vs']    = max_vs
    return [DS,CS,VS]

def worker_prepare_prediction_samples(out_dir,samples,seqs,sv_types,sids,cluster_sids,cluster_over):
    DS,CS,VS = {},{},{}
    for sample_path in sorted(samples): #can do this in ||
        sample = sample_path.split('/')[-1]
        VS[sample] = vcf_to_vcam(sample_path,out_dir,sids,seqs,sv_types,verbose=True)
        CS[sample] = vcam_to_clust(VS[sample],out_dir,seqs,cluster_sids,over=cluster_over,verbose=True)
        DS[sample] = vcam_clust_to_chord(VS[sample],CS[sample],out_dir)[sample]
        CS[sample] = CS[sample][sample]
        VS[sample] = VS[sample][sample]
    for svtype in sv_types: #check to ensure all callers are harmonized
        min_vs,max_vs = get_min_max_vs(DS,svtype)
        if min_vs!=max_vs: #expansion is needed-----------------------------------
            max_vs_idx = {max_vs[i]: i for i in range(len(max_vs))} #expand to max
            for sm in DS:
                if svtype in DS[sm]:
                    for seq in DS[sm][svtype]:
                        vs = DS[sm][svtype][seq]['vs']
                        vs_idx = {vs[i]:i for i in range(len(vs))}
                        idx_vs = {i:vs[i] for i in range(len(vs))}
                        if len(vs)<len(max_vs):
                            chord = DS[sm][svtype][seq]['chord']
                            clean = np.zeros((chord.shape[0],len(max_vs),chord.shape[2]),dtype=chord.dtype)+np.nan
                            for i in range(len(chord)):
                                for j in idx_vs:
                                    clean[i][max_vs_idx[idx_vs[j]]] = chord[i][j]
                            DS[sm][svtype][seq]['chord'] = clean
                            DS[sm][svtype][seq]['vs']    = max_vs
    return [DS,CS,VS]

def process_prepare_training_samples(out_dir,partitions,seqs,sv_types,sids,cluster_sids,cluster_over=0.75,true_over=0.5,cpus=12):
    global result_list
    p2 = mp.Pool(processes=cpus)
    for samples in partitions: #each will be 1 to n number of samples to process one at a time, #partitions is || factor
        p2.apply_async(worker_prepare_training_samples,
                       args=(out_dir,samples,seqs,sv_types,sids,cluster_sids,cluster_over,true_over),
                       callback=collect_results)
        time.sleep(0.5)
    p2.close()
    p2.join()

    DS,CS,VS = {},{},{}
    for result in result_list:
        ds,cs,vs = result
        for sm in ds: DS[sm] = ds[sm]
        for sm in cs: CS[sm] = cs[sm]
        for sm in vs: VS[sm] = vs[sm]
    result_list = []
    return DS,CS,VS

