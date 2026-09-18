#!/usr/bin/env python3
#microhomology (MH) breakpoint features for reSolVer, edited and made by Josiah Rondon
#(c) 2022-2024 Timothy James Becker
#
#MH is scored from the reference at each caller's own breakpoints, so it enters the chord
#matrix as one additional per caller feature next to center,span,geno. naming follows the
#prototype notebooks: read_fasta_region, find_microhomology_robust, mh_length, del_len, chrom.
#
#measured effect, hgsv2 GRCh38, 23 samples, 23 sequences, DEL, 352,660 clusters of which
#100,443 reciprocally overlap the long read truth set. paired on cluster coordinates, one
#HistGradientBoosting fit per arm, split by sample, 20 random splits, nothing selected on the
#test data:
#    center,span,geno            f1 0.8012 +/- 0.0028   AUC 0.9444
#    + mh_score + km_score       f1 0.8196 +/- 0.0014   AUC 0.9594
#    paired delta                f1 +0.0184, 95% CI [+0.0172,+0.0196], MH ahead on 20/20 splits
#the baseline reproduces resolver's own reported DEL f1 of 0.7980 from a full --iterations 30
#run, which is what says the protocol is measuring the same thing. splitting the two features:
#km_score alone is +0.0170 and mh_score alone is +0.0031, so the shared k-mer measure carries
#most of it and the exact run measure still adds a little on top of it.
#
#multiprocessing notes (https://docs.python.org/3/library/multiprocessing.html):
#  * Pool workers are daemonic and a daemonic process cannot create child processes, so MH is
#    scored INSIDE the sample level pool resolver already runs, never in a nested Pool of its
#    own. parallel_enrich_vcam_files below is the standalone entry point for use outside it.
#  * under 'spawn' (win32/macOS default) module globals are re-initialised per worker, under
#    'fork' they are inherited along with open file handles. _REFERENCES is therefore keyed by
#    pid so a forked child never shares the parent's file object and its seek offset.
#  * only the FASTA path crosses the process boundary, never the sequence itself, which keeps
#    the pickled arguments small. each worker opens its own reader lazily on first use, which
#    is the same effect as Pool(initializer=...) without changing resolver's pool setup.

import os
import re
import glob
import gzip
import time
import bisect
import pickle
import numpy as np
import multiprocessing as mp

try:    import pysam           #optional, only for a bgzipped reference
except ImportError: pysam = None

MH_FLANK     = 10           #homology arm cap: MMEJ microhomology runs 2-10bp, so mh_length is 0..20
MH_CONTEXT   = 500          #bp of context each side that the background model samples over
MH_TOL       = 2            #bp of breakpoint imprecision allowed when reading the homology.
                            #measured on hgsv2/GRCh38, DEL, truth matching vs not, within each
                            #caller (Cohen's d): at tol=0 Delly2 shows -0.003 and at tol=2 it
                            #shows +0.270, because tol=0 scores WHERE a caller puts its
                            #breakpoint inside a homology run, which is a convention (Manta
                            #left normalizes and so reads the whole run, mean 5.65 vs 0.8 for
                            #everyone else) rather than whether the run is there at all. the
                            #signal saturates by tol=2
#a vcam call becomes [start,end,geno,ins,info,mh_length,mh_score,km_shared,km_score]
MH_POS       = 5            #mh_length,  raw bp of exact breakpoint homology
MH_SCORE_POS = 6            #mh_score,   [0,1] background normalized
KM_POS       = 7            #km_shared,  raw count of k-mers the two neighborhoods share
KM_SCORE_POS = 8            #km_score,   [0,1] background normalized
CHORD_POS    = [MH_SCORE_POS,KM_SCORE_POS]  #the two the chord carries, raw stays for diagnostics
BASE_LEN     = 5            #a call before any enrichment: [start,end,geno,ins,info]
MH_SVTYPES   = ['DEL','DUP','INV','INS']
COMP         = {'A':'T','C':'G','G':'C','T':'A','N':'N'}

#the background is conditioned on how big the event is, the size classes from the whiteboard
SIZE_BINS = [(50,100),(100,500),(500,1000),(1000,10000),(10000,100000),(100000,int(1e6))]

def size_bin(svlen):
    #index of the size class an event falls in, clamped at both ends
    for i,(lo,hi) in enumerate(SIZE_BINS):
        if svlen<hi: return i
    return len(SIZE_BINS)-1

#:::::::::::::::::::::::::::::::: reference access ::::::::::::::::::::::::::::::::
class ReferenceReader:
    """random access into an indexed FASTA

    the prototype cached whole chromosomes in _CHROMOSOME_SEQUENCES, which costs ~250MB per
    chromosome per worker. a breakpoint only ever needs 2*flank bases, so this reads through
    the .fai instead and keeps one small block cached, which is what makes MH affordable to
    run for every call of every caller. pysam is used when importable, otherwise a pure python
    faidx reader takes over so MH still works where pysam has no wheel (win32)"""
    def __init__(self,fasta_path,block=int(1<<16)):
        self.path   = fasta_path
        self.block  = block
        self.fai    = {}  #chrom -> (length,offset,linebases,linewidth)
        self.alias  = {}  #chr20/20 style name harmonization -> fai key
        self.handle = None
        self.fa     = None
        self._c_chrom,self._c_start,self._c_buf = None,0,b'' #one block, kept as raw bytes so
                                                             #only the slice a call asks for
                                                             #pays for decoding
        self.hits,self.reads = 0,0
        if fasta_path.endswith('.gz') or fasta_path.endswith('.bgz'):
            if pysam is None:
                raise IOError('bgzipped reference needs pysam, use an uncompressed FASTA: %s'%fasta_path)
            self.fa = pysam.FastaFile(fasta_path)
            for i,name in enumerate(self.fa.references): self.fai[name] = (self.fa.lengths[i],None,None,None)
        else:
            self.fai    = self.load_fai(fasta_path)
            self.handle = open(fasta_path,'rb')
        for name in self.fai:
            for a in self.name_variants(name): self.alias[a] = name

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def close(self):
        if self.handle is not None: self.handle.close(); self.handle = None
        if self.fa is not None:     self.fa.close();     self.fa = None

    @staticmethod
    def name_variants(name):
        n = name.split()[0]
        if n.startswith('chr'): return [n,n[3:]]
        return [n,'chr%s'%n]

    @staticmethod
    def load_fai(fasta_path):
        fai_path,fai = fasta_path+'.fai',{}
        if not os.path.exists(fai_path): fai_path = ReferenceReader.build_fai(fasta_path)
        if fai_path is not None and os.path.exists(fai_path):
            with open(fai_path,'r') as f:
                for line in f:
                    row = line.rstrip('\n').split('\t')
                    if len(row)>=5: fai[row[0].split()[0]] = (int(row[1]),int(row[2]),int(row[3]),int(row[4]))
        else: fai = ReferenceReader.scan_fai(fasta_path)
        return fai

    @staticmethod
    def scan_fai(fasta_path):
        #one linear pass that records the faidx tuple of every chromosome
        fai,name,offset,length,linebases,linewidth = {},None,0,0,0,0
        with open(fasta_path,'rb') as f:
            pos = 0
            for raw in f:
                if raw.startswith(b'>'):
                    if name is not None: fai[name] = (length,offset,linebases,linewidth)
                    name = raw[1:].decode('utf-8').split()[0]
                    offset,length,linebases,linewidth = pos+len(raw),0,0,0
                else:
                    n = len(raw.rstrip(b'\r\n'))
                    if linebases==0: linebases,linewidth = n,len(raw)
                    length += n
                pos += len(raw)
            if name is not None: fai[name] = (length,offset,linebases,linewidth)
        return fai

    @staticmethod
    def build_fai(fasta_path):
        """samtools faidx compatible index, written next to the FASTA when that is writable

        several pool workers can reach this at once on a fresh reference, so the index is
        written under a pid unique name and moved into place: a reader then sees either no
        index or a whole one, never half of one"""
        fai_path,tmp_path = fasta_path+'.fai',fasta_path+'.%s.fai.tmp'%os.getpid()
        try:
            fai = ReferenceReader.scan_fai(fasta_path)
            with open(tmp_path,'w') as f:
                for name in fai: f.write('%s\t%s\t%s\t%s\t%s\n'%((name,)+fai[name]))
            os.replace(tmp_path,fai_path)
            return fai_path
        except (IOError,OSError):
            try: os.remove(tmp_path)
            except OSError: pass
            return None

    def key(self,chrom):
        if chrom in self.fai:   return chrom
        if chrom in self.alias: return self.alias[chrom]
        return None

    def length(self,chrom):
        k = self.key(chrom)
        if k is None: return 0
        return self.fai[k][0]

    def fetch(self,chrom,start,stop):
        #0-based half open, clipped to the chromosome, upper case, '' when chrom is unknown
        k = self.key(chrom)
        if k is None: return ''
        n = self.fai[k][0]
        start,stop = max(0,start),min(stop,n)
        if stop<=start: return ''
        if self._c_chrom==k and self._c_start<=start and stop<=self._c_start+len(self._c_buf):
            self.hits += 1
            return self._c_buf[start-self._c_start:stop-self._c_start].decode('ascii').upper()
        b_start = max(0,start-self.block//4)
        b_stop  = min(n,b_start+self.block)
        if b_stop-b_start<stop-start: b_start,b_stop = start,stop
        self._c_chrom,self._c_start,self._c_buf = k,b_start,self.read(k,b_start,b_stop)
        self.reads += 1
        return self._c_buf[start-b_start:stop-b_start].decode('ascii').upper()

    def read(self,k,start,stop):
        #returns the block with its line breaks stripped, still as bytes
        if self.fa is not None: return self.fa.fetch(k,start,stop).encode('ascii')
        length,offset,linebases,linewidth = self.fai[k]
        b1 = offset+start//linebases*linewidth+start%linebases
        b2 = offset+(stop-1)//linebases*linewidth+(stop-1)%linebases+1
        self.handle.seek(b1)
        return self.handle.read(b2-b1).translate(None,b'\r\n')

_REFERENCES = {} #keyed by (pid,path): a forked worker must not inherit the parent file handle
                 #or the two of them would fight over one shared seek offset
def load_reference(fasta_path):
    """the per process ReferenceReader, opened lazily on first use inside each pool worker"""
    k = (os.getpid(),fasta_path)
    if k not in _REFERENCES: _REFERENCES[k] = ReferenceReader(fasta_path)
    return _REFERENCES[k]

def close_references():
    pid = os.getpid()
    for k in [k for k in _REFERENCES if k[0]==pid]: _REFERENCES.pop(k).close()

def read_fasta_region(fasta_path,chrom,start,stop):
    """same signature as the prototype, but indexed rather than whole chromosome in memory"""
    return load_reference(fasta_path).fetch(chrom,start,stop)

#:::::::::::::::::::::::::::::::: homology scoring ::::::::::::::::::::::::::::::::
def match_vector(ref,chrom,l,r,svtype='DEL',ins_seq=None,flank=MH_FLANK):
    """pair up the two breakpoint neighborhoods of one call and return where they agree

    l,r bound the affected segment so that it is ref[l:r]. offset i of the returned vector
    holds the comparison that has to hold for the segment to slide by i: ref[l+i]==ref[r+i]
    for DEL/DUP, the complement of ref[r-1-i] for the inverted junction of an INV, and the
    inserted base ins_seq[i] for an INS. index zero is the reported breakpoint, so an unbroken
    run of agreement across zero is the homology that made the junction possible. N never
    agrees since it carries no homology signal. returns (vector,zero)"""
    if svtype=='INS':
        if ins_seq is None or len(ins_seq)<1: return [],0
        ins_seq,M = ins_seq.upper(),[]
        w         = min(flank,len(ins_seq))
        off,b     = max(0,l-w),ref.fetch(chrom,l-w,l+w)
        for i in range(-w,w):
            a,bi = ins_seq[i],l+i-off
            M += [0<=bi<len(b) and a==b[bi] and a!='N']
        return M,w
    l_off,la = max(0,l-flank),ref.fetch(chrom,l-flank,l+flank)
    r_off,ra = max(0,r-flank),ref.fetch(chrom,r-flank,r+flank)
    M = []
    for i in range(-flank,flank):
        ai = l+i-l_off
        bi = (r-1-i-r_off if svtype=='INV' else r+i-r_off)
        if 0<=ai<len(la) and 0<=bi<len(ra):
            a = la[ai]
            b = (COMP.get(ra[bi],'N') if svtype=='INV' else ra[bi])
            M += [a==b and a!='N']
        else: M += [False]
    return M,flank

def straddle(M,zero,tol=0):
    """longest run of agreement that a breakpoint placed within tol of zero can sit inside

    a run spanning vector indices [i,j) lets the junction sit at any offset in [i-zero,j-zero],
    so with tol=0 this is the ambiguity of the reported breakpoint exactly (the same quantity
    delly and manta report as HOMLEN) and a larger tol reads the homology of the locus through
    imprecise breakpoints"""
    best,i,n = 0,0,len(M)
    while i<n:
        if M[i]:
            j = i
            while j<n and M[j]: j += 1
            if i-zero<=tol and j-zero>=-tol: best = max(best,j-i)
            i = j
        else: i += 1
    return best

def find_microhomology(ref,chrom,start,end,svtype,ins_seq=None,flank=MH_FLANK,tol=0):
    """mh_length of one vcam call, 0..2*flank

    start,end are the VCF POS and END, so POS is the padding base ahead of the event and the
    affected segment is ref[start:end] in 0-based slice form"""
    if svtype=='INS': l,r = start,start
    else:             l,r = start,end
    if svtype!='INS' and r-l<1: return 0
    M,zero = match_vector(ref,chrom,l,r,svtype,ins_seq,flank)
    return straddle(M,zero,tol)

#:::::::::::::::::::::::::: approximate homology, shared k-mers ::::::::::::::::::::::::::
#find_microhomology needs an UNBROKEN run of exact matches, which is the right model for MMEJ
#but structurally blind to NAHR: Alu elements are 85-95% identical to one another, so an
#Alu-Alu junction hits a mismatch within 10-20bp and the exact run stops there no matter how
#far the flank is allowed to reach. measured on hgsv2/GRCh38 only 0.2% of true deletions carry
#100bp or more of exact homology, and raising MH_FLANK from 10 to 1000 lowered the within
#caller separation rather than raising it.
#counting k-mers the two breakpoint neighborhoods have in common tolerates those mismatches: a
#6-mer survives wherever 6 consecutive bases agree, so two 90% identical elements still share
#a large fraction of their k-mers. this is the measure the prototype notebooks reach for
#(kmer_count_fast, shared_kmers, build_kmer_background in res_mh.py / vcf_copy.py)

#swept on hgsv2/GRCh38 over k in (5,6,7) and window in (50,150), both pairings, scoring by
#within caller Cohen's d: k=7 window=150 aligned won at mean |d| 0.664 against 0.155 for the
#exact run measure, and the two are nearly uncorrelated (pearson r 0.002 to 0.112)
KM_K      = 7    #k-mer size. the chance expectation for two windows of n positions each is
                 #about n*n/4**k shared, so k sets how much of the score is noise
KM_WINDOW = 150  #bp each side of a breakpoint that the k-mer sets are drawn from
KM_SCALE  = 8.0  #km_score is log2(count+1), so ~0..8: divides it onto the other columns' scale

def kmer_count_fast(seq,k):
    """count k-mers in a string, skipping any that contain N. returns {kmer:count}"""
    if len(seq)<k: return {}
    counts = {}
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        if 'N' in kmer: continue
        counts[kmer] = counts.get(kmer,0)+1
    return counts

def shared_kmers(ref,chrom,start,end,k=KM_K,window=KM_WINDOW,mode='aligned'):
    """how many distinct k-mers the two breakpoint neighborhoods have in common

    mode='aligned' takes a window centred on each breakpoint. that is the pairing NAHR
    produces: if the junction fell at offset t inside two homologous elements, then the
    sequence around start and the sequence around end are the same part of the two copies.
    mode='junction' takes the prototype's outside flanks, ref[start-w:start] against
    ref[end:end+w], which are the two pieces that become adjacent once the segment is gone.
    windows are shrunk so they can never overlap, which would otherwise compare a region with
    itself on short events"""
    w = window
    if end>start: w = max(k,min(w,(end-start)//2))
    if mode=='junction': r1,r2 = ref.fetch(chrom,start-w,start),ref.fetch(chrom,end,end+w)
    else:                r1,r2 = ref.fetch(chrom,start-w,start+w),ref.fetch(chrom,end-w,end+w)
    a,b = kmer_count_fast(r1,k),kmer_count_fast(r2,k)
    if len(a)<1 or len(b)<1: return 0
    return len(set(a).intersection(set(b)))

def find_microhomology_robust(seq,del_len,flank=MH_FLANK,tol=0):
    """prototype signature: score a deletion straight off a window sequence

    seq is a window centred on the deletion and del_len its length, as in the notebooks. this
    shares the same run of agreement core as find_microhomology, so both report the same
    mh_length for the same junction"""
    half = len(seq)//2
    l,r  = half,half+del_len
    if del_len<1 or r>len(seq): return 0
    M = []
    for i in range(-flank,flank):
        ai,bi = l+i,r+i
        if 0<=ai<len(seq) and 0<=bi<len(seq): M += [seq[ai]==seq[bi] and seq[ai]!='N']
        else:                                 M += [False]
    return straddle(M,flank,tol)

#::::::::::::::::::::::::::: chromosome arms and background :::::::::::::::::::::::::::
#a raw homology length is not comparable across the genome: 4bp of agreement is unremarkable
#in pericentromeric repeat, and surprising in unique sequence. so MH is normalized against a
#background sampled from the same chromosome arm and the same event size class, which is what
#MH[seq][arm] and the size bins on the design board describe. arms come from the assembly's
#own N runs: the telomere gaps bound the chromosome and the largest interior gap is the
#centromere, so no external cytoband file is needed.

def scan_n_runs(ref,chrom,min_gap=10000,block=int(1<<22)):
    """every run of N at least min_gap long, as [(start,stop),...] in 0-based half open"""
    n,runs,run_start,pos = ref.length(chrom),[],None,0
    while pos<n:
        s = ref.fetch(chrom,pos,min(n,pos+block))
        if len(s)<1: break
        for m in re.finditer('N+',s):
            a,b = pos+m.start(),pos+m.end()
            if run_start is not None and a==run_start[1]: run_start = (run_start[0],b)
            else:
                if run_start is not None and run_start[1]-run_start[0]>=min_gap: runs += [run_start]
                run_start = (a,b)
        pos += len(s)
    if run_start is not None and run_start[1]-run_start[0]>=min_gap: runs += [run_start]
    return runs

def find_arms(ref,chrom,min_gap=10000):
    """p and q arm bounds of one chromosome, taken from its N runs

    returns {'p':(start,stop),'q':(start,stop)}. an assembly that models its centromere
    instead of leaving a gap (some GRCh38 releases) yields one arm named 'q' spanning the
    whole sequence, which is still a valid unit to normalize within"""
    n     = ref.length(chrom)
    runs  = scan_n_runs(ref,chrom,min_gap)
    lo,hi = 0,n
    if len(runs)>0 and runs[0][0]<=min_gap:    lo = runs[0][1]     #leading telomere gap
    if len(runs)>0 and runs[-1][1]>=n-min_gap: hi = runs[-1][0]    #trailing telomere gap
    interior = [r for r in runs if r[0]>lo and r[1]<hi]
    if len(interior)<1: return {'q':(lo,hi)}
    cen = sorted(interior,key=lambda r: r[1]-r[0])[-1]             #largest gap is the centromere
    return {'p':(lo,cen[0]),'q':(cen[1],hi)}

class MHBackground:
    """empirical distribution of mh_length per (chrom, arm, svtype, size bin)

    built by sampling random events of the right size inside the arm and scoring them the same
    way a real call is scored, so the null carries the arm's real repeat content rather than an
    i.i.d. base composition. mh_score is then the mid rank position of an observation in that
    distribution, in [0,1]: 0 means the homology is entirely ordinary for this arm and size,
    near 1 means it is longer than almost anything the arm produces by chance"""
    def __init__(self,fasta_path,flank=MH_FLANK,context=MH_CONTEXT,n_samples=500,seed=1,tol=0):
        self.fasta_path = fasta_path
        self.flank      = flank
        self.tol        = tol
        self.context    = context
        self.n_samples  = n_samples
        self.seed       = seed
        self.arms       = {}   #chrom -> {'p':(a,b),'q':(a,b)}
        self.dist       = {}   #(chrom,arm,svtype,bin) -> sorted [mh_length,...]

    def path(self):
        #tol is in the name: a null has to be built the same way the observations are scored
        return '%s.mhbg.f%s.t%s.k%s.w%s.n%s.pickle'%(self.fasta_path,self.flank,self.tol,
                                                      KM_K,KM_WINDOW,self.n_samples)

    def arm_of(self,chrom,pos):
        for arm,(a,b) in self.arms.get(chrom,{}).items():
            if a<=pos<b: return arm
        return 'q'

    def build(self,chroms,svtypes=('DEL','INV'),verbose=False):
        import random as _random
        ref = load_reference(self.fasta_path)
        for chrom in chroms:
            if ref.key(chrom) is None: continue
            t0 = time.time()
            self.arms[chrom] = find_arms(ref,chrom)
            rnd = _random.Random('%s:%s'%(self.seed,chrom))
            for arm,(a_,b_) in self.arms[chrom].items():
                for svtype in svtypes:
                    for bi,(lo,hi) in enumerate(SIZE_BINS):
                        vals,tries = [],0
                        while len(vals)<self.n_samples and tries<self.n_samples*8:
                            tries += 1
                            svlen  = rnd.randint(lo,hi-1)
                            if b_-a_<=svlen+2*self.context: break
                            s_ = rnd.randint(a_+self.context,b_-self.context-svlen)
                            if ref.fetch(chrom,s_,s_+self.flank).count('N')>0: continue
                            if ref.fetch(chrom,s_+svlen-self.flank,s_+svlen).count('N')>0: continue
                            vals += [find_microhomology(ref,chrom,s_,s_+svlen,svtype,None,self.flank,self.tol)]
                        self.dist[(chrom,arm,svtype,bi)] = sorted(vals)
                    #the k-mer measure shares the window pairing across svtypes, so one null
                    #covers them all and it is only sampled once per (arm,size bin)
                    if svtype==svtypes[0]:
                        kvals,tries = [],0
                        while len(kvals)<self.n_samples and tries<self.n_samples*8:
                            tries += 1
                            svlen  = rnd.randint(lo,hi-1)
                            if b_-a_<=svlen+2*self.context: break
                            s_ = rnd.randint(a_+self.context,b_-self.context-svlen)
                            if ref.fetch(chrom,s_,s_+self.flank).count('N')>0: continue
                            kvals += [shared_kmers(ref,chrom,s_,s_+svlen,KM_K,KM_WINDOW)]
                        self.dist[(chrom,arm,'KM',bi)] = sorted(kvals)
            if verbose:
                print('mh: background for %s arms=%s in %.1fs'%(chrom,','.join(sorted(self.arms[chrom])),time.time()-t0))
        return self

    def score(self,chrom,start,svlen,svtype,mh_length):
        """mid rank of mh_length in its background, in [0,1]. 0.0 when there is no background"""
        sv = ('INV' if svtype=='INV' else 'DEL')     #DEL/DUP/INS all share the direct walk
        d  = self.dist.get((chrom,self.arm_of(chrom,start),sv,size_bin(svlen)))
        if not d: return 0.0
        lo = bisect.bisect_left(d,mh_length)
        hi = bisect.bisect_right(d,mh_length)
        return (lo+hi)/(2.0*len(d))                  #mid rank avoids saturating at 0 or 1

    def score_kmer(self,chrom,start,svlen,km_shared):
        """km_score is log2(km_shared+1), a plain monotone rescale, NOT a background rank

        both normalizations were tried on hgsv2/GRCh38 and both lost signal. within caller
        Cohen's d, averaged over callers: raw count 0.619, arm+size mid rank 0.361, log2
        enrichment over the null mean 0.234. the reason is that the null samples uniformly
        inside an arm, so it is dominated by satellite and segmental duplication that no
        caller ever calls into, and every real call sits in its low tail. the null is simply
        not representative of callable sequence, so dividing or ranking by it discards the
        variation that matters. log2 keeps the ordering exactly, which is all a tree needs,
        and bounds the range for the MLP. the arm and size context the board asks for is still
        available to the model: span is already a chord feature. self and svlen are kept in
        the signature so a callable-region null can be swapped in later without a churn"""
        return float(np.log2(km_shared+1.0))

    def save(self):
        tmp = self.path()+'.%s.tmp'%os.getpid()
        try:
            with gzip.GzipFile(tmp,'wb') as f:
                pickle.dump({'arms':self.arms,'dist':self.dist,'flank':self.flank,'tol':self.tol,
                             'context':self.context,'n_samples':self.n_samples,'seed':self.seed},f)
            os.replace(tmp,self.path())
            return self.path()
        except (IOError,OSError):
            try: os.remove(tmp)
            except OSError: pass
            return None

    def load(self):
        if not os.path.exists(self.path()): return False
        try:
            with gzip.GzipFile(self.path(),'rb') as f: d = pickle.load(f)
        except (IOError,OSError,EOFError,pickle.UnpicklingError): return False
        self.arms,self.dist = d['arms'],d['dist']
        return True

_BACKGROUNDS = {}   #keyed by (pid,path) like the readers, built once and reused per worker
def load_background(fasta_path,chroms=None,flank=MH_FLANK,context=MH_CONTEXT,n_samples=500,
                    seed=1,build=True,verbose=False,tol=0):
    """the per process MHBackground, read from its cache file or built and cached on first use"""
    k = (os.getpid(),fasta_path,flank,tol,n_samples)
    if k in _BACKGROUNDS: return _BACKGROUNDS[k]
    bg = MHBackground(fasta_path,flank,context,n_samples,seed,tol)
    bg.load()
    if chroms is None: chroms = sorted(load_reference(fasta_path).fai)
    #a cache file only covers the sequences it was built over. loading one that was written by
    #a smaller run would leave every other chromosome scoring 0 with no warning, so the
    #sequences that are missing are built and merged in rather than trusted blindly
    ref     = load_reference(fasta_path)
    missing = [c for c in chroms if c not in bg.arms and ref.key(c) is not None]
    if len(missing)>0:
        if not build:
            if len(bg.arms)<1: return None
            print('mh: WARNING cached background covers %s sequences, missing %s'%(len(bg.arms),len(missing)))
            _BACKGROUNDS[k] = bg
            return bg
        if verbose: print('mh: building background over %s sequences (%s already cached) -> %s'%                          (len(missing),len(bg.arms),os.path.basename(bg.path())))
        bg.build(missing,verbose=verbose)
        bg.save()
    _BACKGROUNDS[k] = bg
    return bg

#:::::::::::::::::::::::::::::::: vcam enrichment ::::::::::::::::::::::::::::::::
def vcam_has_mh(V):
    #True when the call records of a vcam already carry mh_length and mh_score
    for sample in V:
        for caller_id in V[sample]:
            for svtype in V[sample][caller_id]:
                for chrom in V[sample][caller_id][svtype]:
                    for call in V[sample][caller_id][svtype][chrom]: return len(call)>KM_SCORE_POS
    return True

def mh_work_units(V,sv_types,flank,tol):
    """split a vcam into one work unit per (caller_id,svtype,chrom)

    a unit carries only the coordinates and, for INS, the inserted sequence, so what crosses
    a process boundary stays small. calls that already hold an mh_length are left out"""
    keys,units = [],[]
    for sample in V:
        for caller_id in sorted(V[sample]):
            for svtype in sorted(V[sample][caller_id]):
                if svtype not in sv_types: continue
                for chrom in sorted(V[sample][caller_id][svtype]):
                    todo = [(c[0],c[1],c[3]) for c in V[sample][caller_id][svtype][chrom]
                            if len(c)<=KM_SCORE_POS]
                    if len(todo)>0:
                        keys  += [(sample,caller_id,svtype,chrom)]
                        units += [(chrom,svtype,todo,flank,tol)]
    return keys,units

def worker_mh_lengths(fasta_path,chrom,svtype,calls,flank,tol,background=True):
    """score one work unit, module level so Pool can pickle it. returns [(length,score),...]

    the worker opens its own ReferenceReader on first use, so the FASTA path is all that is
    sent across, never the sequence. the background is read from its cache file the same way"""
    ref = load_reference(fasta_path)
    bg  = (load_background(fasta_path,flank=flank,tol=tol,build=False) if background else None)
    out = []
    for c in calls:
        svlen     = abs(c[1]-c[0])
        mh_length = find_microhomology(ref,chrom,c[0],c[1],svtype,c[2],flank,tol)
        km_shared = shared_kmers(ref,chrom,c[0],c[1],KM_K,KM_WINDOW)
        out += [(mh_length,0.0 if bg is None else bg.score(chrom,c[0],svlen,svtype,mh_length),
                 km_shared,0.0 if bg is None else bg.score_kmer(chrom,c[0],svlen,km_shared))]
    return out

def enrich_vcam_with_mh(V,fasta_path,sv_types=None,flank=MH_FLANK,tol=0,n_jobs=1,verbose=False,
                        background=True):
    """append mh_length and mh_score to every call of every caller of one sample vcam, in place

    mh_length is the raw homology in bp and stays in the vcam for diagnostics. mh_score is that
    length placed in the background distribution of its own chromosome arm and size class, and
    it is the value the chord carries. background=False leaves mh_score at 0.

    n_jobs>1 shards the sample by (caller_id,svtype,chrom) over a Pool. that is only legal
    when this process is not itself a Pool worker, since a daemonic process cannot create
    children, so a daemonic caller silently falls back to the serial walk. inside resolver
    that is exactly what happens: scaling there comes from the sample level pool that is
    already running, and every one of its workers scores its own samples in parallel"""
    if sv_types is None: sv_types = MH_SVTYPES
    if n_jobs>1 and not mp.current_process().daemon:
        return parallel_enrich_vcam_with_mh(V,fasta_path,sv_types,flank,tol,n_jobs,verbose,background)
    ref,n,missing = load_reference(fasta_path),0,set([])
    #the background is only ever READ here, never built. a pool worker that built its own would
    #save a model covering just its own chromosomes and clobber a fuller one, so building is the
    #caller's job: resolver main does it before the pool starts, parallel_enrich_vcam_files does
    #it before its own
    bg = (load_background(fasta_path,flank=flank,tol=tol,build=False) if background else None)
    if background and bg is None:
        print('mh: WARNING no background cached for %s, mh_score will be 0. '%fasta_path+
              'call mh.load_background(ref,chroms) once before enriching')
    for sample in V:
        for caller_id in sorted(V[sample]):
            for svtype in sorted(V[sample][caller_id]):
                for chrom in sorted(V[sample][caller_id][svtype]):
                    if ref.key(chrom) is None: missing.add(chrom)
                    score = svtype in sv_types
                    for call in V[sample][caller_id][svtype][chrom]:
                        if len(call)>KM_SCORE_POS: continue   #already enriched
                        #a vcam written by an older feature set carries some of the values but
                        #not all of them, so drop what is there rather than appending past it
                        if len(call)>BASE_LEN: del call[BASE_LEN:]
                        svlen = abs(call[1]-call[0])
                        if score:
                            mh_length = find_microhomology(ref,chrom,call[0],call[1],svtype,call[3],flank,tol)
                            km_shared = shared_kmers(ref,chrom,call[0],call[1],KM_K,KM_WINDOW)
                        else: mh_length,km_shared = 0,0
                        call.append(mh_length)
                        call.append(0.0 if bg is None else bg.score(chrom,call[0],svlen,svtype,mh_length))
                        call.append(km_shared)
                        call.append(0.0 if bg is None else bg.score_kmer(chrom,call[0],svlen,km_shared))
                        n += 1
    if verbose:
        print('mh: scored %s calls for sample=%s (%s reference reads, %s cached%s)'%\
              (n,sorted(V)[0],ref.reads,ref.hits,('' if bg is not None else ', no background')))
        if len(missing)>0: print('mh: WARNING %s not found in %s, scored as 0'%(','.join(sorted(missing)),fasta_path))
    return V

def parallel_enrich_vcam_with_mh(V,fasta_path,sv_types=None,flank=MH_FLANK,tol=0,n_jobs=4,
                                 verbose=False,background=True):
    """the || form of enrich_vcam_with_mh for one sample, sharded by chromosome and caller

    a worker cannot mutate the parent's vcam, so each returns its (mh_length,mh_score) pairs
    and the parent writes them back in the order the unit was built. svtypes outside sv_types
    never become a unit and are filled in with zeros here"""
    if sv_types is None: sv_types = MH_SVTYPES
    keys,units = mh_work_units(V,sv_types,flank,tol)
    scored     = {}
    if len(units)>0:
        args      = [(fasta_path,)+u+(background,) for u in units]
        chunksize = max(1,len(args)//(n_jobs*4))
        with mp.Pool(processes=n_jobs) as pool:
            for k,pairs in zip(keys,pool.starmap(worker_mh_lengths,args,chunksize=chunksize)):
                scored[k] = pairs
    n = 0
    for sample in V:
        for caller_id in sorted(V[sample]):
            for svtype in sorted(V[sample][caller_id]):
                for chrom in sorted(V[sample][caller_id][svtype]):
                    got,i = scored.get((sample,caller_id,svtype,chrom),None),0
                    for call in V[sample][caller_id][svtype][chrom]:
                        if len(call)>KM_SCORE_POS: continue
                        if len(call)>BASE_LEN: del call[BASE_LEN:]
                        if got is None: call.extend([0,0.0,0,0.0])
                        else:           call.extend(list(got[i])); i += 1
                        n += 1
    if verbose: print('mh: scored %s calls for sample=%s over %s units on %s jobs'%(n,sorted(V)[0],len(units),n_jobs))
    return V

#:::::::::::::::::::::::::::: standalone || enrichment ::::::::::::::::::::::::::::
def worker_enrich_vcam_file(vcam_path,fasta_path,sv_types,flank,tol):
    """enrich one already written .vcam.gz in place, module level so it stays picklable"""
    with gzip.GzipFile(vcam_path,'rb') as f: V = pickle.load(f)
    if vcam_has_mh(V): return (vcam_path,0)
    enrich_vcam_with_mh(V,fasta_path,sv_types,flank,tol)
    with gzip.GzipFile(vcam_path,'wb') as f: pickle.dump(V,f)
    close_references()
    n = sum([len(V[sm][c][sv][sq]) for sm in V for c in V[sm] for sv in V[sm][c] for sq in V[sm][c][sv]])
    return (vcam_path,n)

def parallel_enrich_vcam_files(vcam_glob,fasta_path,sv_types=None,flank=MH_FLANK,tol=0,n_jobs=4,verbose=True):
    """add MH to a folder of vcams, one worker per sample: this is the axis that scales

    a sample is the right unit of work. sharding a single sample instead (n_jobs on
    enrich_vcam_with_mh) has to stand up a Pool per sample, and under the spawn start method
    that startup costs more than the scoring, so it only pays off on fork or for one very
    large sample. resolver itself gets its MH parallelism this way, from the sample level
    pool it already runs in process_prepare_training_samples.

    for use OUTSIDE resolver's pipeline only. a Pool worker is daemonic and cannot create
    child processes, so this must never be called from inside another Pool. the caller needs
    the usual if __name__=='__main__' guard for the spawn start method"""
    if sv_types is None: sv_types = MH_SVTYPES
    args = [(p,fasta_path,sv_types,flank,tol) for p in sorted(glob.glob(vcam_glob))]
    if len(args)<1: return []
    load_background(fasta_path,flank=flank,tol=tol,verbose=verbose) #built here, in the parent so the
                                                             #workers only ever read the cache
    with mp.Pool(processes=n_jobs) as pool:
        results = pool.starmap(worker_enrich_vcam_file,args,chunksize=1)
    if verbose:
        for path,n in results: print('mh: %s calls scored in %s'%(n,path))
    return results
