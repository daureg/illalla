photos.svg: photos.json photos.css
	kartograph photos.json -o photos.svg -s photos.css 
	sed -i 's/^.*<\/metadata>//' photos.svg
	sed -i s/'r="2"/r="0.7"/g' photos.svg

combine: sf_map.svg photos.svg
	cat sf_map.svg goldengatepark.svg photos.svg > sf.svg #CHANGE
	rsvg-convert --background-color='#fff' -a -z 1.5 -d 96 -p 96 sf.svg -o sf.png
