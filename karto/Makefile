sf.svg: sf.json style.css
	kartograph sf.json -o sf.svg -s style.css 
	sed -i 's/r="2"/r="1"/g' sf.svg
	rsvg-convert -a -z 2.5 -d 200 -p 200 sf.svg -o sf.png

usa.svg: test.json style.css
	kartograph test.json -o usa.svg -s style.css 
	sed -i 's/r="2"/r="1"/g' usa.svg
	rsvg-convert -a -z 3 -d 200 -p 200 usa.svg -o usa.png
