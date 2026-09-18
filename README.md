<img alt="GitHub Downloads (all assets, all releases)" src="https://img.shields.io/github/downloads/timothyjamesbecker/reSolVer/total"> [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)<br>
![Alt text](resolver_diagram.png?raw=true "reSolVer")
<br>(c) 2022-2024 Timothy James Becker
## Resolving multiple structural variation callers and platforms by matrix transformation and imputation

## Requirements (non-docker)
3.11+ python <br>
1.26+ numpy <br>
0.22+ pysam (optional, only needed for a bgzipped --ref) <br>
1.2+  scikit-learn <br>

## Installation
pip is not setup at this time, simply clone the repo and then use the resolver.py script to run model training, testing and prediciton as described below in the usage section.

## Usage
```bash
resolver.py \
--in_dirs dir1,dir2,dir3... \
--pred_dirs dirP \
--out_dir dirP_result \
--seqs chr1,chr2,... \
--split 0.5
--iterations 30 \
--sv_types DEL,DUP,INV,INS \
--ref /path/hg38.fa \
--cpus 12
```

<b>--in_dirs:</b> comma seperated list of stage-mapped SV sample folders. Each sample folder should contain SV calls that have simplified naming conventions: samplename_SID. For example sample HG00096 would have the Lumpy SV file named according to the stage ID as HG00096_S18.vcf. True calls that are given from 1000 Genomes or other projects act as supervision for the machine learning have the unique stage ID as 9. For example HG00096_S0.vcf is the true set of SV calls for sample HG00096.<br>
<br>
<b>--pred_dir:</b> comma seperated directors to run prediction on. This would be where you would put new calls that you wanted to run reSolVer models on that were pretrained such as the resolver_g1kp3 or resolver_multi models.<br>
<br>
<b>--out_dir:</b> output directory where the chrod,clust,models,train,vcam,vcfs folders and files will be generated.<br>
<br>
<b>--seqs:</b> sequences that the training,testing,and validation (or prediction) should be perfromed on. From standard VCF calling you would want all the reference chromosomes of interest.<br>
<br>
<b>--split:</b> used for training to set the amount of data that will be given to the modeling during the fitting or training phase. The remaining proportion will then be split into the testing phase to estimate the f1 accuracy and then the final amount which are validation are never seen by the model or estimated. They can be used to produce an unbiased final model estimate during the prediction once the models parameters are fixed or to calibrate the probabilty performance.<br>
<br>
<b>--iterations:</b> the number of iterations(random tries) that the modeling will use to fit during training. Higher iterations will often overfit and low will underfit to the training portion of the data.<br>
<br>
<b>--sv_types:</b> comma seperated list of SV types, can be used to build models for every type or just for one at a time in case of || training.<br>
<br>
<b>--cpus:</b> the number of computer cores to use for data processing and model fiting<br>
<br>
<b>--ref:</b> reference FASTA that turns on the two sequence features, microhomology and shared k-mers. Optional: leave it out and reSolVer behaves exactly as before. When it is given, every call of every caller is scored at its own breakpoints and the chord grows from three per caller columns to five. Measured on hgsv2/GRCh38 this is worth <b>+0.0184 DEL f1</b>, 0.8012 to 0.8196, 95% CI [+0.0172,+0.0196], ahead on 20 of 20 sample splits. See the Microhomology section below for the protocol and the per feature split. The FASTA is read through a samtools compatible .fai index, built next to it on first use if missing, and chr20/20 style naming is harmonized against the VCF. Because the features change the width of the chord, these runs cache into vcam, chord, train and models files tagged <b>.mh</b> and never mix with the plain ones, so both can live in one --out_dir.<br>
<br>
<b>--mh_flank:</b> bp scanned on each side of a breakpoint, which also caps each homology arm (default 10, the MMEJ scale). The background model samples over a wider ±500bp of context regardless<br>
<br>
<b>--mh_tol:</b> bp of breakpoint imprecision to allow when reading MH, default 2. At 0 the score depends on where inside a homology run a caller chose to put its breakpoint, which is a convention rather than biology: Manta left normalizes its deletions and so reads the whole run (mean mh_length 5.65 on GRCh38) while every other caller sits near 0.8. Measured within each caller on hgsv2/GRCh38, truth matching calls versus that same caller's false ones, Delly2 separates at Cohen's d=-0.003 at tol=0 and +0.270 at tol=2, and the signal saturates there. 2 is therefore the default<br>

## Microhomology (mh.py)
![Alt text](resolver_mh_diagram.png?raw=true "reSolVer microhomology")

Two sequence features are read off the reference at each caller's own breakpoints and added to the chord next to center, span and genotype. Both are optional: without <b>--ref</b> reSolVer behaves exactly as before.

<b>mh_score</b> is microhomology proper. When a double strand break is repaired by microhomology mediated end joining (MMEJ) the two ends anneal over a few bases and one copy is lost, so the reference is left holding the same few bases at both breakpoints of the resulting deletion. A caller that put its breakpoints on a real junction lands on that homology; a caller that guessed does not.

<b>km_score</b> is how many k-mers the two breakpoint neighborhoods have in common. Exact microhomology is blind to NAHR, because Alu elements are only 85-95% identical to one another and an unbroken run of exact matches stops at the first mismatch. Counting shared k-mers tolerates those mismatches, and in practice it measures how repetitive the locus is. That turns out to be the stronger of the two signals, because false calls cluster in repeat dense, poorly mappable sequence.

### Results
Measured on hgsv2 against GRCh38 (hg38.no_alt), 23 samples, 23 sequences, DEL, 352,660 clusters of which 100,443 reciprocally overlap the long read (PAV) truth set. The two arms are paired on cluster coordinates, one HistGradientBoosting fit per arm, split by sample, 20 random splits, nothing selected on the test data. <b>w1</b> is the weighted_f1 that worker_train_classifier actually selects on, shown at a precision leaning and a recall leaning f1_w:

| metric | baseline | with MH | paired &Delta; | 95% CI |
| --- | --- | --- | --- | --- |
| <b>f1</b> | <b>0.8012</b> | <b>0.8196</b> | <b>+0.0184</b> | <b>[+0.0172, +0.0196]</b> |
| w1 (f1_w=2.0) | 0.8257 | 0.8436 | +0.0179 | [+0.0169, +0.0190] |
| w1 (f1_w=0.5) | 0.7782 | 0.7968 | +0.0187 | [+0.0158, +0.0216] |
| precision | 0.8800 | 0.8963 | +0.0163 | [+0.0106, +0.0221] |
| recall | 0.7362 | 0.7551 | +0.0188 | [+0.0130, +0.0247] |
| PR-AUC | 0.9049 | 0.9230 | +0.0181 | [+0.0179, +0.0183] |
| ROC-AUC | 0.9444 | 0.9594 | +0.0150 | [+0.0149, +0.0151] |

MH is ahead on 20 of 20 splits and every 95% CI excludes zero. Precision and recall both rise, so this is not a threshold shift trading one against the other. PR-AUC is quoted alongside ROC-AUC because at 28.5% positives ROC-AUC is insensitive to the class imbalance in a way that flatters the numbers; the two agree here. The baseline arm reproduces reSolVer's own reported DEL f1 of 0.7980 from a full <b>--iterations 30</b> run, which is what says the protocol is measuring the same quantity rather than a different one.

Splitting the two features apart, by f1:

| features | f1 | paired &Delta; f1 | 95% CI |
| --- | --- | --- | --- |
| center, span, geno | 0.8012 | | |
| + mh_score | 0.8036 | +0.0031 | [+0.0025, +0.0036] |
| + km_score | 0.8176 | +0.0170 | [+0.0155, +0.0185] |
| + both | 0.8196 | +0.0184 | [+0.0172, +0.0196] |

The shared k-mer measure carries most of it and the exact run measure still adds a little on top.

Two caveats worth stating. This is DEL only: DUP produced no truth intersecting clusters in hgsv2 at this overlap threshold and INV had 60 test positives, so neither is measurable here. And it is one reference and one truth set, so it is evidence that the features work, not that the size of the effect generalizes.

The gain is not spread evenly. Splitting clusters by how many callers support them, MH adds +0.0215 AUC on single caller clusters and +0.0366 on clusters with four or more, against +0.0080 in the middle. That matters because <b>84.6% of clusters carry only one or two callers, and half of all true SVs live there</b>: caller concordance, which is reSolVer's main strength, has the least to say about exactly those calls.

### Where it sits
```
sample_SID.vcf(.gz)  --[vcf_reader]->  VCAM  [start,end,geno,ins,info]
                                      |
        reference.fa + .fai --> enrich_vcam_with_mh <-- <ref>.mhbg.pickle
                                      v                 (arm + size bin nulls)
                       VCAM  [ ...,info,mh_length,mh_score,km_shared,km_score]
                                      |
                              vcam_to_clust (start,end only: neither feature moves a cluster)
                                      v
                            build_cluster_matrix(mh_i=CHORD_POS)
                                      v
           chord  [callers x (center,span,geno,mh_score,km_score)]  <- 3 wide without --ref
                                      v
                    IterativeImputer --> AdaBoost / HistGB / MLP
```
All four values are computed once per sample, at VCF read time, and then travel inside the vcam pickle. Nothing recomputes them: clustering, the chord, training and prediction all carry what is already there. The raw `mh_length` and `km_shared` stay in the vcam for diagnostics while the chord carries the two scores. Runs that use a reference write their vcam, chord, train and models files with a <b>.mh</b> tag, so a 3 wide model can never be loaded against a 5 wide chord.

### Naming conventions
The module keeps the names used in the prototype notebooks (vcf_copy.py, vcam.py, res_mh.py) so the two read the same way:

<b>chrom</b> a reference sequence name, harmonized across chr20/20 spelling<br>
<b>arm</b> 'p' or 'q', the unit a score is normalized within<br>
<b>del_len</b> the length of a deletion, as in the notebooks<br>
<b>mh_length</b> raw bp of exact breakpoint homology, appended at MH_POS<br>
<b>mh_score</b> that length ranked in its own background, [0,1], at MH_SCORE_POS<br>
<b>km_shared</b> raw count of k-mers the two neighborhoods share, at KM_POS<br>
<b>km_score</b> log2(km_shared+1), at KM_SCORE_POS<br>
<b>CHORD_POS</b> the two positions the chord carries, so nothing downstream hardcodes an index<br>
<b>caller_id</b> the stage id (SID) of a caller, the key of a vcam<br>
<b>ins_seq</b> an inserted sequence, used to score INS<br>
<b>flank / tol</b> bp scanned each side of a breakpoint, and the breakpoint imprecision allowed<br>
<b>read_fasta_region(fasta_path,chrom,start,stop)</b> the prototype's region accessor, backed here by a .fai index instead of a whole chromosome held in memory<br>
<b>kmer_count_fast(seq,k)</b> the prototype's k-mer counter, N bearing k-mers skipped<br>
<b>find_microhomology_robust(seq,del_len)</b> the prototype's signature, scoring a deletion straight off a window sequence<br>
<b>find_microhomology(ref,chrom,start,end,svtype)</b> the same score taken from the indexed reference, which is what the pipeline calls<br>

### How the two scores are computed
For a call bounding the segment ref[start:end], offset <b>i</b> of a match vector records the comparison that would have to hold for that segment to slide by i: ref[start+i]==ref[end+i] for DEL and DUP, the complement of ref[end-1-i] for the inverted junction of an INV, and ins_seq[i] for an INS. An unbroken run of agreement across offset zero is the homology the junction sits on, and its length is `mh_length` - the same quantity delly and manta report as HOMLEN. A deletion of d bases inside a homopolymer run of length L scores exactly L-d, because it can be placed anywhere in the run. N never counts as agreement.

`km_shared` counts the distinct k-mers common to a window centred on each breakpoint, k=7 over 150bp, chosen by sweeping k over (5,6,7) and the window over (50,150) and scoring by within caller separation. Windows shrink so they can never overlap on a short event.

<b>--mh_tol</b> matters more than it looks. At 0 the score turns on where inside a homology run a caller chose to place its breakpoint, which is a convention rather than biology: Manta left normalizes its deletions and so reads the whole run, giving a mean `mh_length` of 5.65 on real GRCh38 sequence while every other caller sits near 0.8. Measured within each caller, Delly2 separates truth matching from false calls at Cohen's d = -0.003 at tol=0 and +0.270 at tol=2, because tol=2 asks whether the homology is there at all rather than who lined up with it. The signal saturates by 2, which is the default.

<b>--mh_flank</b> is capped at 10, the MMEJ scale, and raising it makes things worse rather than better: at flank 1000 the within caller separation falls from 0.155 to 0.113, because exact matching cannot reach NAHR homology anyway and the extra range only lets false calls accumulate longer chance runs. Catching NAHR is what km_score is for.

### The arm background
Chromosome arms come from the assembly's own N runs, so no cytoband file is needed: the leading and trailing gaps are the telomeres and the largest interior gap is the centromere. An assembly that models its centromere instead of leaving a gap yields a single arm spanning the sequence, which is still a valid unit to normalize within.

For each (chrom, arm, svtype, size bin) a null distribution is built by sampling random events of the right size inside that arm and scoring them the way a real call is scored. Size bins are 50-100, 100-500, 500-1K, 1K-10K, 10K-100K and 100K-1M. `mh_score` is the mid rank of an observation in that null, in [0,1].

`km_score` deliberately does <b>not</b> use it. Within caller separation, averaged over callers: raw count 0.619, arm and size mid rank 0.361, log2 enrichment over the null mean 0.234. The null samples uniformly inside an arm, so it is dominated by satellite and segmental duplication that no caller ever calls into, and every real call lands in its low tail. The null is not representative of callable sequence, so ranking or dividing by it discards the variation that matters. `km_score` is therefore log2(km_shared+1), which keeps the ordering exactly.

The background is built once per reference and cached next to it as `<ref>.mhbg.f<flank>.t<tol>.k<k>.w<window>.n<samples>.pickle`, about 5s per chromosome and 1.7 min for a whole genome. Every parameter that changes the null is in the file name, because a null has to be built the same way the observations are scored. reSolVer builds it in `__main__` before the sample pool starts, and `load_background` checks that a cached file actually covers the sequences being asked for, filling in any that are missing rather than silently scoring them 0.

### Parallelism
Per the multiprocessing docs a Pool worker is daemonic and a daemonic process cannot create child processes, so MH never opens a Pool of its own inside reSolVer. It is scored inside the sample level pool that `process_prepare_training_samples` already runs, and that is where the scaling comes from: one worker per sample, each opening its own reference reader on first use. Only the FASTA path crosses a process boundary, never sequence. Measured on 8 samples, ~87k calls each pass: 7.9s at one job, 3.1s at two (2.05x), 2.5s at four (3.14x), 2.4s at eight, where it becomes I/O bound on the reference and the pickles.

Two entry points exist outside the pipeline. <b>parallel_enrich_vcam_files(glob,ref,n_jobs)</b> adds the features to a folder of vcams that were built before they existed, one worker per sample, building the background in the parent first and skipping any file already scored. <b>enrich_vcam_with_mh(...,n_jobs=N)</b> shards a single sample by (caller_id,svtype,chrom); it has to stand up a Pool per sample, so under the spawn start method its startup costs about as much as the work, and it is worth using on fork or for one very large sample. Both detect a daemonic caller and fall back to the serial walk rather than failing.

### Cost
A breakpoint needs only 2*flank bases for the exact run and 2*window for the k-mers, so the reference is read through its .fai a block at a time rather than held in memory a chromosome at a time as the prototype did. Scoring every call of every caller for one sample costs roughly a second per four chromosomes, around 11,000 calls per second per core, and is lost in the noise of the VCF read it rides along with. The vcam, chord and train files grow by a few percent. The cost that is not small is imputation: the chord is two thirds wider, so IterativeImputer has that much more to fit.
