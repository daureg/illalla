TAG=$1
ssh -t lefalg1@kosh.aalto.fi "export LD_LIBRARY_PATH=/u/99/lefalg1/unix/lib; cd ../data/flickr/illalla; python plot_tag.py $TAG; cp ${TAG}_1.* disc/; cd disc; sed -i 's/src\": \"[^_]\+_1/src\": \"${TAG}_1/' dots.json; make dots.svg"
scp lefalg1@kosh.aalto.fi:/u/99/lefalg1/data/flickr/illalla/disc/dots.svg disc/${TAG}.svg
