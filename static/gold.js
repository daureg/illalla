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
	for (var i = 0, l = features.length; i < l; i++) {
		console.log(features[i].geometry);
		poly = geojson_to_polygon(features[i].geometry);
		if (bounds) {bounds.extend(poly.getBounds());}
		else {bounds = poly.getBounds();}
		GOLD.addLayer(poly);
	}
    map.fitBounds(bounds, {maxZoom: 13});
}
$(function() {
	map.addLayer(GOLD);
    $.request('get', $SCRIPT_ROOT+'/static/ground_truth.json')
	.then(function get_gt(result) {
		var areas = $.parseJSON(result);
		plot_gold(areas[DISTRICT].gold[CITY]);
	})
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
});
