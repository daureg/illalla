/* return the LatLng in the middle of an array of LatLng */
function barycenter(points) {
    var lat = 0, lng = 0, n = points.length;
    for (var i = n - 1; i >= 0; i--) {
        lat += points[i].lat;
        lng += points[i].lng;
    }
    return new L.LatLng(lat/n, lng/n);
}

function create_map(div_id, center, main_layer, bbox) {
    var offset = 0.01;
    var opacity = full_d ? 0.5 : 1;
    var southWest = new L.LatLng(bbox[0][0] - offset, bbox[0][1] - offset);
    var northEast = new L.LatLng(bbox[2][0] + offset, bbox[2][1] + offset);
    var bounds = new L.LatLngBounds(southWest, northEast);
    var map = new L.Map(div_id, {zoom: 12, minZoom: 11, center: center,
                                 layers: [main_layer.setOpacity(opacity)], maxBounds: bounds})
        .fitBounds(bounds);
    L.polygon(bbox, {fill: false, weight: 2}).addTo(map);
    return map;
}
var OpenStreetMap_Mapnik = L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        {attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'});
var OpenStreetMap_HOT = L.tileLayer('http://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
	attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Tiles courtesy of <a href="http://hot.openstreetmap.org/" target="_blank">Humanitarian OpenStreetMap Team</a>'
});
var center = new L.LatLng(0.5*(bbox[0][0]+bbox[2][0]),
                          0.5*(bbox[0][1]+bbox[2][1]));
map = create_map('map', center, OpenStreetMap_Mapnik, bbox);

// photos means checkins as background
var COLS = {'photos': '#000', 'checkins': '#f00',
    'only_photos': '#0f0', 'only_checkins': '#00f'};
function read_discrepancies(what) {
    $http({
        'url': city + '_' + what + '_d.json',
        'method': "GET",
        'good': function (req) {
            console.log("success");
            if (what === 'full') {
                plot_full($json(req.responseText));
            } else {
                plot_discrepancies(what, $json(req.responseText));
            }
        },
        'error': function (req) {
            console.log("error");
            console.log(req.responseText);
        }
    });
}
function how_many_more(what, ph, ch) {
    if (what === 'only_photos' || what == 'only_checkins') {
        return '';
    }
    if (what === 'checkins') {
        local_ratio = photos_ratio / (ph / ch);
    }
    else {
        local_ratio = checkins_ratio / (ch / ph);
    }
    return '<br>'+local_ratio.toFixed(1)+' more ' + what + ' than usual.';
}
function plot_full(discrepancies) {
	L.geoJson(discrepancies, {
		style: function (feature) {
			var c = feature.properties.color;
			return {fillColor: c, fillOpacity: 0.7, weight: 0};
		},
		// onEachFeature: function (feature, layer) {
		// 	var r = feature.properties.ratio.toPrecision(3);
		// 	layer.bindLabel(r);
		// }
	}).addTo(map);
}
function plot_discrepancies(what, discrepancies) {
    bcolor = COLS[what];
	L.geoJson(discrepancies, {
		style: function (feature) {
			var c = feature.properties.color;
			return {fillColor: c, fillOpacity: 0.4,
				color: bcolor, weight: 4, opacity: 0.6};
		},
		onEachFeature: function (feature, layer) {
            layer.on({click: zoomToFeature});
			var p = feature.properties;
			var d = p.discrepancy.toPrecision(5);
			var ph = p.photos.toFixed(0);
			var ch = p.checkins.toFixed(0);
            var more = how_many_more(what, ph, ch);
			layer.bindLabel(d+'<br>photos: '+ph+'<br>checkins: '+ch+more);
		}
	}).addTo(map);
}
function zoomToFeature(e) { map.fitBounds(e.target.getBounds()); }
if (full_d) {
	read_discrepancies('full');
} else {
	read_discrepancies('checkins');
	read_discrepancies('photos');
	read_discrepancies('only_photos');
	read_discrepancies('only_checkins');
}
