cd ..
time python2 spatial_scan.py $1
cd disc
sed -i "/CHANGE/ s/map.svg [^.]\+\./map.svg ${1}./" Makefile
make combine
mv sf.png ${1}.png
firefox ${1}.png
