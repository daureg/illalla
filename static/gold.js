// http://stackoverflow.com/a/2880929
var urlParams;
(window.onpopstate = function () {
    var match,
    pl     = /\+/g,  // Regex for replacing addition symbol with a space
    search = /([^&=]+)=?([^&]*)/g,
    decode = function (s) { return decodeURIComponent(s.replace(pl, " ")); },
    query  = window.location.search.substring(1);
    urlParams = {};
    while (match = search.exec(query))
        urlParams[decode(match[1])] = decode(match[2]);
})();
var VENUES_LOC = {};
var GOLD = new L.FeatureGroup();
var LEFT_CANVAS = null, LC_VISIBLE = false;
var map = create_map('map', BBOX, {zoomAnimation: false});
var left = map;
// populate('left', canvas_display);
var MyLayer = L.FullCanvas.extend({
    drawSource: function(point, ctx) {
        ctx.beginPath();
        ctx.fillStyle = "rgba(33, 33, 33, .72)";
        ctx.arc(point.x, point.y , 2.0, 0, 2 * Math.PI, true);
        ctx.fill();
    }
});
function canvas_display(result, nside, map) {
    var venues = $.parseJSON(result).r;
    var points = [];
    var marker = null;
    _.each(venues, function add_venue(venue) {
        VENUES_LOC[venue._id] = venue.loc;
        var d = {"slat": venue.loc[0], "slon": venue.loc[1]};
        points.push(d);
    });
    var venue_dots = new MyLayer();
    venue_dots.setData(points);
	LEFT_CANVAS = venue_dots;
	LC_VISIBLE = true;
    map.addLayer(venue_dots);
}
function geojson_to_polygon(geo, pstyle) {
	var style = pstyle || {color: '#b22222', opacity: 0.6};
	if (geo.type === 'Polygon') {
		var coords = geo.coordinates[0], latlngs = [];
		for (var i = 0, l = coords.length; i < l-1; i++) {
			latlngs.push([coords[i][1], coords[i][0]]);
		}
		return L.polygon(latlngs, style);
	}
	return L.circle([geo.center[0], geo.center[1]], geo.radius, style);
}
function plot_gold(features) {
	var poly = null,
		bounds = null;
	GOLD.clearLayers();
	for (var i = 0, l = features.length; i < l; i++) {
		poly = geojson_to_polygon(features[i].geometry);
		if (bounds) {bounds.extend(poly.getBounds());}
		else {bounds = poly.getBounds();}
		GOLD.addLayer(poly);
	}
    map.fitBounds(bounds, {maxZoom: 12});
}
function plot_answers(features, metric) {
	var poly = null;
	for (var i = 0, l = features.length; i < l; i++) {
		if (features[i].metric === metric && features[i].pos < 4) {
			console.log(features[i].geo);
			poly = geojson_to_polygon(features[i].geo, {color: '#5677fc', opacity: 0.6});
			// console.log(poly);
			GOLD.addLayer(poly);
		}
	}
}
var AREAS = null;
$(function() {
	console.log(urlParams);
	map.addLayer(GOLD);
    $.request('get', $SCRIPT_ROOT+'/static/ground_truth.json')
	.then(function get_gt(result) {
		AREAS = $.parseJSON(result);
		plot_gold(AREAS[DISTRICT].gold[CITY]);
	})
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
	if (urlParams.m === 'emd') {
		$.request('get', $SCRIPT_ROOT+'/static/cmp_'+urlParams.from+'.json')
		.then(function get_emd(result) {
			AREAS = $.parseJSON(result);
			plot_answers(AREAS[CITY][DISTRICT], 'emd');
		});
	}
	if (urlParams.m === 'femd') {
		$.request('get', $SCRIPT_ROOT+'/static/'+urlParams.from+'_'+DISTRICT+'_'+CITY+'.json')
		.then(function get_emd(result) {
			AREAS = $.parseJSON(result);
			plot_answers(AREAS, 'femd');
		});
	}
// {65: 'A', 66: 'B', 67: 'C', 68: 'D', 69: 'E', 70: 'F', 71: 'G', 72: 'H',
// 73: 'I', 74: 'J', 75: 'K', 76: 'L', 77: 'M', 78: 'N', 79: 'O', 80: 'P', 81:
// 'Q', 82: 'R', 83: 'S', 84: 'T', 85: 'U', 86: 'V', 87: 'W', 88: 'X', 89:
// 'Y', 90: 'Z'}
	document.addEventListener('keydown', function(event) {
		var infos = {
			71: 'triangle', // G olden triangle
			76: 'latin', // L atin
			77: 'montmartre', // M ontmartre
			80: 'pigalle', // P igalle
			82: 'marais', // Ma R ais
			79: 'official', // O fficials
			72: '16th', // 16t H
			87: 'weekend', // W eekend
		};
		var district = infos[event.keyCode];
		if (district) {plot_gold(AREAS[district].gold[CITY]);}
	});
});
