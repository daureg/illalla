photos.svg: photos.json photos.css
	kartograph photos.json -o photos.svg -s photos.css 
	sed -i 's/^.*<\/metadata>//' photos.svg
	sed -i s/'r="2"/r="0.7"/g' photos.svg

combine: sf_map.svg photos.svg
	cat sf_map.svg photos.svg > sf.svg
	rsvg-convert --background-color='#fff' -a -z 3 -d 200 -p 200 sf.svg -o sf.png

usa.svg: test.json style.css
	kartograph test.json -o usa.svg -s style.css 
	sed -i 's/r="2"/r="1"/g' usa.svg
	rsvg-convert -a -z 3 -d 200 -p 200 usa.svg -o usa.png

sf_map.svg: sf.json map.css
	kartograph sf.json -o sf_map.svg -s map.css 
	sed -i 's/<\/svg>//' sf_map.svg
