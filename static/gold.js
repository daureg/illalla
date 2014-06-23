var VENUES_LOC = {};
var GOLD = new L.FeatureGroup();
var LEFT_CANVAS = null, LC_VISIBLE = false;
var map = create_map('map', BBOX, {zoomAnimation: false});
var left = map;
populate('left', canvas_display);
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
function geojson_to_polygon(geo) {
	var style = {color: '#b22222', opacity: 0.6};
	if (geo.type === 'Polygon') {
		var coords = geo.coordinates[0], latlngs = [];
		for (var i = 0, l = coords.length; i < l-1; i++) {
			latlngs.push([coords[i][1], coords[i][0]]);
		}
		return L.polygon(latlngs, style);
	}
	return L.circle([geo.center[1], geo.center[0]], geo.radius, style);
}
function plot_gold(features) {
	var poly = null,
		bounds = null;
	GOLD.clearLayers();
	for (var i = 0, l = features.length; i < l; i++) {
		console.log(features[i].geometry);
		poly = geojson_to_polygon(features[i].geometry);
		if (bounds) {bounds.extend(poly.getBounds());}
		else {bounds = poly.getBounds();}
		GOLD.addLayer(poly);
	}
    map.fitBounds(bounds, {maxZoom: 14});
}
var AREAS = null;
$(function() {
	map.addLayer(GOLD);
    $.request('get', $SCRIPT_ROOT+'/static/ground_truth.json')
	.then(function get_gt(result) {
		console.log('get something');
		AREAS = $.parseJSON(result);
		plot_gold(AREAS[DISTRICT].gold[CITY]);
	})
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
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
