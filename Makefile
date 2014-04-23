SOURCES_JS = static/minified-src.js \
	     static/leaflet-src.js
SOURCES_CSS = static/normalize.css \
	      static/leaflet.css
PROD = 0
NODE=../node_modules

all: app.css app.js

app.css: $(SOURCES_CSS)
	cat $(SOURCES_CSS) > __tmp.css
ifeq ($(PROD), 1)
	$(NODE)/clean-css/bin/cleancss -e --s0 __tmp.css > $@
else
	cat __tmp.css > $@
endif
	rm __tmp.css
	mv app.css static/app.css

app.js: $(SOURCES_JS)
ifeq ($(PROD), 1)
	# sed -e '/console.log(/d' $(SOURCES_JS) > __tmp.js
	# $(NODE)/uglify-js/bin/uglifyjs __tmp.js -cm > $@
	sed -e '/console.log(/d' static/cnn.js > __tmp.js
	$(NODE)/uglify-js/bin/uglifyjs __tmp.js -cm > static/rcnn.js
	sed -i  "s/='cnn.js'/='rcnn.js'/" templates/cnn.html
	rm __tmp.js
else
	$(NODE)/uglify-js/bin/uglifyjs $(SOURCES_JS) -b > $@
	sed -i  "s/='rcnn.js'/='cnn.js'/" templates/cnn.html
endif
	# mv app.js static/app.js

clean:
	rm -f static/app.css* static/app.js*
