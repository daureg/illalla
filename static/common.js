var MINI = require('minified');
var _=MINI._, $=MINI.$, $$=MINI.$$, EE=MINI.EE, HTML=MINI.HTML;
L.Icon.Default.imagePath = '/static/images';
var ratio = 2/5;
var length = parseInt(41*ratio),
    width = parseInt(25*ratio),
    middle = parseInt(12.5*ratio);
L.Icon.Small = L.Icon.Default.extend({
    options: {
        iconSize: [width, length],
        shadowSize: [length, length],
        iconAnchor: [middle, length],
    }});
var smallIcon = new L.Icon.Small();

/* return the LatLng in the middle of an array of LatLng */
function barycenter(points) {
    var lat = 0, lng = 0, n = points.length;
    for (var i = n - 1; i >= 0; i--) {
        lat += points[i].lat || points[i][1];
        lng += points[i].lng || points[i][0];
    }
    return new L.LatLng(lat/n, lng/n);
}

/* Return the LatLngBounds enclosing `bbox` */
function compute_bound(bbox) {
    var offset = 0.01;
    var southWest = new L.LatLng(bbox[0][0] - offset, bbox[0][1] - offset);
    var northEast = new L.LatLng(bbox[2][0] + offset, bbox[2][1] + offset);
    return new L.LatLngBounds(southWest, northEast);
}

function create_map(div_id, bbox, extras) {
	var carto_layer = L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
	{attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'});
    // carto_layer = L.tileLayer('http://{s}.tile.openstreetmap.org/hot/{z}/{x}/{y}.png', {
    // attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Tiles courtesy of <a href="http://hot.openstreetmap.org/" target="_blank">Humanitarian OpenStreetMap Team</a>'});
	if (L.Google) {
		carto_layer = new L.Google('ROADMAP', { mapOptions: { styles: {featureType: 'all'} } });
	}
    var bbounds = compute_bound(bbox);
    var center = new L.LatLng(0.5*(bbox[0][0]+bbox[2][0]),
                              0.5*(bbox[0][1]+bbox[2][1]));
    var options = {zoom: 4, minZoom: 2, center: center, layers:
                  [carto_layer], maxBounds: bbounds};
    _.extend(options, extras);
    var map = new L.Map(div_id, options)
        .fitBounds(bbounds);
    // L.polygon(bbox, {fill: false, weight: 3}).addTo(map);
    return map;
}

function populate(side, display) {
    var origin = null, mymap = null, nside = null;
    if (side === 'left') {nside = 0; origin = true; mymap = left;}
    else {nside = 1; origin = false; mymap = right;}
    $.request('post', $SCRIPT_ROOT+'/populate', {origin: origin})
    .then(function success(result){
        display(result, nside, mymap);
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
}
