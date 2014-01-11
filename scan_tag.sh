TAG=$1
ssh -t lefalg1@kosh.aalto.fi "cd ../data/flickr/illalla; python spatial_scan.py $TAG && cd dist && make combine"
scp lefalg1@kosh.aalto.fi:/u/99/lefalg1/data/flickr/illalla/disc/sf.png ${TAG}.png
