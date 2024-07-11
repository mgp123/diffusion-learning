rm -f fid_stats.txt
for generation_set in generations/* 
do
    echo $generation_set >> fid_stats.txt
    python3 -m pytorch_fid  reference_images $generation_set >> fid_stats.txt
done