@HD	VN:1.5	SO:coordinate
@SQ	SN:1	LN:249250621
@SQ	SN:2	LN:243199373
@SQ	SN:3	LN:198022430
@SQ	SN:4	LN:191154276
@SQ	SN:5	LN:180915260
@SQ	SN:6	LN:171115067
@SQ	SN:7	LN:159138663
@SQ	SN:8	LN:146364022
@SQ	SN:9	LN:141213431
@SQ	SN:10	LN:135534747
@SQ	SN:11	LN:135006516
@SQ	SN:12	LN:133851895
@SQ	SN:13	LN:115169878
@SQ	SN:14	LN:107349540
@SQ	SN:15	LN:102531392
@SQ	SN:16	LN:90354753
@SQ	SN:17	LN:81195210
@SQ	SN:18	LN:78077248
@SQ	SN:19	LN:59128983
@SQ	SN:20	LN:63025520
@SQ	SN:21	LN:48129895
@SQ	SN:22	LN:51304566
@SQ	SN:X	LN:155270560
@SQ	SN:Y	LN:59373566
@SQ	SN:MT	LN:16569
@SQ	SN:GL000207.1	LN:4262
@SQ	SN:GL000226.1	LN:15008
@SQ	SN:GL000229.1	LN:19913
@SQ	SN:GL000231.1	LN:27386
@SQ	SN:GL000210.1	LN:27682
@SQ	SN:GL000239.1	LN:33824
@SQ	SN:GL000235.1	LN:34474
@SQ	SN:GL000201.1	LN:36148
@SQ	SN:GL000247.1	LN:36422
@SQ	SN:GL000245.1	LN:36651
@SQ	SN:GL000197.1	LN:37175
@SQ	SN:GL000203.1	LN:37498
@SQ	SN:GL000246.1	LN:38154
@SQ	SN:GL000249.1	LN:38502
@SQ	SN:GL000196.1	LN:38914
@SQ	SN:GL000248.1	LN:39786
@SQ	SN:GL000244.1	LN:39929
@SQ	SN:GL000238.1	LN:39939
@SQ	SN:GL000202.1	LN:40103
@SQ	SN:GL000234.1	LN:40531
@SQ	SN:GL000232.1	LN:40652
@SQ	SN:GL000206.1	LN:41001
@SQ	SN:GL000240.1	LN:41933
@SQ	SN:GL000236.1	LN:41934
@SQ	SN:GL000241.1	LN:42152
@SQ	SN:GL000243.1	LN:43341
@SQ	SN:GL000242.1	LN:43523
@SQ	SN:GL000230.1	LN:43691
@SQ	SN:GL000237.1	LN:45867
@SQ	SN:GL000233.1	LN:45941
@SQ	SN:GL000204.1	LN:81310
@SQ	SN:GL000198.1	LN:90085
@SQ	SN:GL000208.1	LN:92689
@SQ	SN:GL000191.1	LN:106433
@SQ	SN:GL000227.1	LN:128374
@SQ	SN:GL000228.1	LN:129120
@SQ	SN:GL000214.1	LN:137718
@SQ	SN:GL000221.1	LN:155397
@SQ	SN:GL000209.1	LN:159169
@SQ	SN:GL000218.1	LN:161147
@SQ	SN:GL000220.1	LN:161802
@SQ	SN:GL000213.1	LN:164239
@SQ	SN:GL000211.1	LN:166566
@SQ	SN:GL000199.1	LN:169874
@SQ	SN:GL000217.1	LN:172149
@SQ	SN:GL000216.1	LN:172294
@SQ	SN:GL000215.1	LN:172545
@SQ	SN:GL000205.1	LN:174588
@SQ	SN:GL000219.1	LN:179198
@SQ	SN:GL000224.1	LN:179693
@SQ	SN:GL000223.1	LN:180455
@SQ	SN:GL000195.1	LN:182896
@SQ	SN:GL000212.1	LN:186858
@SQ	SN:GL000222.1	LN:186861
@SQ	SN:GL000200.1	LN:187035
@SQ	SN:GL000193.1	LN:189789
@SQ	SN:GL000194.1	LN:191469
@SQ	SN:GL000225.1	LN:211173
@SQ	SN:GL000192.1	LN:547496
@SQ	SN:NC_007605	LN:171823
@SQ	SN:hs37d5	LN:35477943
@RG	ID:4CDF14232D_sim_RG	SM:4CDF14232D_sim_	LB:4CDF14232D_sim__LB1	PU:4CDF14232D_sim__RG1	PL:ILLUMINA	CN:NA
@PG	ID:bwa	PN:bwa	VN:0.7.15-r1142-dirty	CL:/software/bwa-master/bwa mem -M -t 8 -R @RG\tID:4CDF14232D_sim_RG\tLB:Solexa4CDF14232D_sim_RG\tPL:illumina\tPU:4CDF14232D_sim_RG\tSM:4CDF14232D_sim_ /home/tbecker/ref/human_g1k_v37_decoy.fa /home/tbecker/4CDF14232D_sim_1.fq.gz /home/tbecker/4CDF14232D_sim_2.fq.gz
@PG	ID:MarkDuplicates	VN:2.5.0(2c370988aefe41f579920c8a6a678a201c5261c1_1466708365)	CL:picard.sam.markduplicates.MarkDuplicates INPUT=[/home/tbecker/4CDF14232D_sim.sorted.bam] OUTPUT=/home/tbecker/4CDF14232D_sim.bam METRICS_FILE=/home/tbecker/4CDF14232D_sim.picard.metrics.txt MAX_RECORDS_IN_RAM=4000000    MAX_SEQUENCES_FOR_DISK_READ_ENDS_MAP=50000 MAX_FILE_HANDLES_FOR_READ_ENDS_MAP=8000 SORTING_COLLECTION_SIZE_RATIO=0.25 REMOVE_SEQUENCING_DUPLICATES=false TAGGING_POLICY=DontTag REMOVE_DUPLICATES=false ASSUME_SORTED=false DUPLICATE_SCORING_STRATEGY=SUM_OF_BASE_QUALITIES PROGRAM_RECORD_ID=MarkDuplicates PROGRAM_GROUP_NAME=MarkDuplicates READ_NAME_REGEX=<optimized capture of last three ':' separated fields as numeric values> OPTICAL_DUPLICATE_PIXEL_DISTANCE=100 VERBOSITY=INFO QUIET=false VALIDATION_STRINGENCY=STRICT COMPRESSION_LEVEL=5 CREATE_INDEX=false CREATE_MD5_FILE=false GA4GH_CLIENT_SECRETS=client_secrets.json	PN:MarkDuplicates
