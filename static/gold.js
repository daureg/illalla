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
var map = create_map('map', BBOX, {zoomAnimation: false, zoomControl: false});
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
function geojson_to_polygon(geo, pstyle) {
	var style = pstyle || {color: '#056f00', opacity: 0.4};
	if (geo.type === 'Polygon') {
		var coords = geo.coordinates[0], latlngs = [];
		for (var i = 0, l = coords.length; i < l-1; i++) {
			latlngs.push([coords[i][1], coords[i][0]]);
		}
		return L.polygon(latlngs, style);
	}
	return L.circle([geo.center[0], geo.center[1]], geo.radius, style);
}
var BOUNDS = null;
var COLORS = ['#e65100', '#5677fc', '#056f00'];
function plot_gold(district_arg) {
	districts = district_arg.split(',');
	// GOLD.clearLayers();
	var poly = null;
	for (var j = 0, len = districts.length; j < len; j++) {
		var features = AREAS[districts[j]].gold[CITY];
		document.getElementById('label').innerHTML += '<span style="color: '+COLORS[j]+'">'+districts[j]+'</span><br>';
		for (var i = 0, l = features.length; i < l; i++) {
			// if (i !== 2) continue;
			// console.log(features[i].properties.nb_venues);
			poly = geojson_to_polygon(features[i].geometry, {opacity: 0.4, color: COLORS[j]});
			if (BOUNDS) {BOUNDS.extend(poly.getBounds());}
			else {BOUNDS = poly.getBounds();}
			GOLD.addLayer(poly);
		}
	}
    map.fitBounds(BOUNDS, {maxZoom: 18});
}
var A_NUM = 1;
function plot_answers(features, metric) {
	var poly = null;
	for (var i = 0, l = features.length; i < l; i++) {
		if (i === 0 && A_NUM ===1) {continue;}
		if (features[i].metric === metric && features[i].pos < 6) {
			poly = geojson_to_polygon(features[i].geo, {color: COLORS[A_NUM], opacity: 0.6});
			BOUNDS.extend(poly.getBounds());
			GOLD.addLayer(poly);
		}
	}
    map.fitBounds(BOUNDS);
	A_NUM += 1;
}
var AREAS = null;
$(function() {
	map.addLayer(GOLD);
    $.request('get', $SCRIPT_ROOT+'/static/ground_truth.json')
	.then(function get_gt(result) {
		AREAS = $.parseJSON(result);
		plot_gold(DISTRICT);
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
		all_from = urlParams.from.split(',');
		console.log(all_from);
		for (var i = 0, len = all_from.length; i < len; i++) {
			console.log($SCRIPT_ROOT+'/static/'+all_from[i]+'_'+DISTRICT+'_'+CITY+'.json');
		$.request('get', $SCRIPT_ROOT+'/static/'+all_from[i]+'_'+DISTRICT+'_'+CITY+'.json')
		.then(function get_emd(result) {
			AREAS = $.parseJSON(result);
			console.table(AREAS);
			plot_answers(AREAS, 'femd');
		});
		}
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
		if (district) {plot_gold(district);}
	});
});
