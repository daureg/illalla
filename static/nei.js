var JUST_READING = false;
var VENUES_LOC = {};
function make_icon(color) {
    var ratio = 4/5;
    var length = parseInt(45*ratio),
        width = parseInt(35*ratio),
        middle = parseInt(17.5*ratio);
    return L.AwesomeMarkers.icon({
        icon: 'home',
        markerColor: color,
        prefix: 'fa',
        // iconSize: [width, length],
        // shadowSize: [length, length],
        // iconAnchor: [middle, length-2],
    });
}
if (JUST_READING) {
    $('#mapl').set('$width', '99.8%');
}
var left = create_map('mapl', LBBOX, {zoomAnimation: false});
var right = create_map('mapr', RBBOX, {zoomAnimation: false});
var map_right_toggle = $('#mapr').toggle({$$fade: 1}, {$$fade: 0}, 150);
populate('left', canvas_display);
if (JUST_READING) {
    map_right_toggle();
}
else {
    populate('right', canvas_display);
}
var VENUE_CARD = '<a href="{{url}}" target="_blank">{{name}}</a>, <small>{{cat}}</small>';
var LEFT_CANVAS = null, LC_VISIBLE = false;
var RIGHT_CANVAS = null, RC_VISIBLE = false;
var VENUES_MARKERS = [new L.FeatureGroup(), new L.FeatureGroup()];
function canvas_display(result, nside, map) {
    var venues = $.parseJSON(result).r;
    var points = [];
    var marker = null;
    _.each(venues, function add_venue(venue) {
        VENUES_LOC[venue._id] = venue.loc;
        var d = {"slat": venue.loc[0], "slon": venue.loc[1]};
        // marker = L.marker(venue.loc, {title: venue.name, icon: smallIcon})
        // .bindPopup(_.formatHtml(VENUE_CARD, venue));
        // VENUES_MARKERS[nside].addLayer(marker);
        points.push(d);
    });
    var venue_dots = new MyLayer();
    venue_dots.setData(points);
    if (nside === 0) {
        LEFT_CANVAS = venue_dots;
        LC_VISIBLE = true;
    }
    else {
        RIGHT_CANVAS = venue_dots;
        RC_VISIBLE = true;
    }
    // map.addLayer(VENUES_MARKERS[nside]);
    map.addLayer(venue_dots);
}
var MyLayer = L.FullCanvas.extend({
    drawSource: function(point, ctx) {
        ctx.beginPath();
        ctx.fillStyle = "rgba(33, 33, 33, .70)";
        ctx.arc(point.x, point.y , 2.25, 0, 2 * Math.PI, true);
        ctx.fill();
    }
});
document.addEventListener('keydown', function(event) {
    if (event.keyCode === 67) {
        if (LC_VISIBLE) {
            left.removeLayer(LEFT_CANVAS);
        }
        else {
            left.addLayer(LEFT_CANVAS);
        }
        LC_VISIBLE = !LC_VISIBLE;
    }
    if (event.keyCode === 68) {
	    map_right_toggle();
    }
    if (event.keyCode === 69) {
        if (RC_VISIBLE) {
            right.removeLayer(RIGHT_CANVAS);
        }
        else {
            right.addLayer(RIGHT_CANVAS);
        }
        RC_VISIBLE = !RC_VISIBLE;
    }
}, false);
var drawnItems = new L.FeatureGroup();
left.addLayer(drawnItems);
var answers = new L.FeatureGroup();
right.addLayer(answers);
var drawControl = new L.Control.Draw({
    position: 'topleft',
    draw: {
        polyline: false,
        marker: false,
        polygon: {
            shapeOptions: { color: '#b22222' },
            allowIntersection: false,
            showArea: false,
        },
        rectangle: {
            shapeOptions: { color: '#1e90ff' },
            showArea: false,
        },
        circle: false
        // {
        //     shapeOptions: { color: '#2ecc40' },
        //     showRadius: false,
        // }
    },
    edit: false
});
left.addControl(drawControl);
left.on('draw:drawstart', function(e) {
	if (e.layerType === 'polygon' && LC_VISIBLE) {
		left.removeLayer(LEFT_CANVAS);
		LC_VISIBLE = false;
	}
});
left.on('draw:created', function (e) {
    drawnItems.clearLayers();
    var type = null, zone = null, lid = null, radius = null, center = null,
        geo = null;
    type = e.layerType;
    zone = e.layer;
    geo = zone.toGeoJSON().geometry;
    radius = 0;
    if (type === 'circle') {
        radius = zone._mRadius;
        center = zone._latlng;
    }
    else {
        type = 'polygon';
        center = barycenter(zone._latlngs);
    }
    drawnItems.addLayer(zone);
    lid = zone._leaflet_id;
    var query = {id: lid, type: type, geo: geo, radius: radius, center: center};
    // $('#log').fill(JSON.stringify(geo))
    // $('.leaflet-draw-section').hide();
    var form_infos = $('#presets').values();
    var metric = form_infos.metric,
	region_name = form_infos.neighborhood;
    left.fitBounds(zone.getBounds(), {maxZoom: 14});
    if (form_infos.candidates == 'full') {
    $.request('post', $SCRIPT_ROOT+'/match_neighborhood',
            {geo: JSON.stringify(geo), metric: metric, name: region_name})
    .then(function success(result){
        if (!JUST_READING) {
            poll_until_done();
        }
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
    }
    else {
        if (!JUST_READING) {
            search_seed(form_infos, zone);
        }
    }
    /*
    */
    if (!LC_VISIBLE) {
	    left.addLayer(LEFT_CANVAS);
	    LC_VISIBLE = true;
    }
});
var RESULT_FMT = 'Smallest distance of {{dst}} for {{nb_venues}} venues in a radius of {{radius}}m.';
var result_circle = null;
function poll_until_done() {
    $.request('get', $SCRIPT_ROOT+'/status', {})
    .then(function success(result) {
        var answer = $.parseJSON(result).r;
        $('#status').set('@value', answer.progress);
        answer.res.dst = answer.res.dst.toFixed(3);
        var radius = answer.res.radius,
        center = answer.res.center;
        $('#res').fill(HTML(RESULT_FMT, answer.res));
        if (center.length === 2) {
            if (result_circle !== null) {
                result_circle.setLatLng(center).setRadius(radius);
            }
            else {
                answers.clearLayers();
                result_circle = L.circle(center, radius, {color: '#2ecc40'});
                answers.addLayer(result_circle);
            }
        }
        if (!answer.done) { poll_until_done(); }
        else {
            output = {};
            output[dest] = {dst: answer.res[0], geo: {
                type: 'circle', center: center, radius: radius}};
            var ans = JSON.stringify(output);
            ans = ans.substr(1, ans.length-2)+',';
            $('#log').fill(ans);
        }
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
}
var TRIANGLE_VENUES = [[
    '4adcdb1ef964a520485f21e3', '4b66e9aaf964a520f02f2be3',
    '4b04208bf964a520835122e3', '4ae9ebcbf964a520a8b721e3',
    '4bd68a9e5631c9b66013a630', '4adcdb24f964a520fb6021e3',
    '4aeea3e4f964a5201bd421e3', '4afaa120f964a5202f1822e3',
    '4bd9c8c62e6f0f472af70b08', '4adcdb24f964a520306121e3',
    '4adcdb24f964a5202c6121e3', '4b83d849f964a5203d1331e3',
    '4b78f93ef964a520dbe72ee3', '4adcdb21f964a5201e6021e3',
    '4c26fcae5c5ca593cb2f47fe', '4adcdb23f964a520ab6021e3',
    '4ba221fbf964a5200ede37e3', '4adcdb21f964a520056021e3',
    '4c097ac53c70b713fdd3275b'],
    ['4adcdb1ef964a5204f5f21e3', '4adcdb1ef964a520585f21e3',
    '4adcdb1ff964a5207b5f21e3', '4adcdb1ff964a520815f21e3',
    '4adcdb1ff964a520855f21e3', '4adcdb1ff964a5208d5f21e3',
    '4adcdb1ff964a520a65f21e3', '4adcdb21f964a520e55f21e3',
    '4adcdb21f964a520eb5f21e3', '4adcdb21f964a520f35f21e3',
    '4adcdb21f964a520f65f21e3', '4adcdb23f964a520af6021e3',
    '4adcdb24f964a520fb6021e3', '4adcdb25f964a520506121e3',
    '4adcdb25f964a5205e6121e3', '4afaa120f964a5202f1822e3',
    '4b07dc1ef964a5208f0023e3', '4b0b8ca3f964a520333223e3',
    '4b0e51faf964a520c75623e3', '4b18032ff964a5206dcb23e3',
    '4b18e708f964a52069d623e3', '4b1aa6ebf964a52068ee23e3',
    '4b253f95f964a520ae6e24e3', '4b2cfb32f964a520bfcb24e3',],[
    '4b59976af964a520f28d28e3', '4b5abb30f964a52061d228e3',
    '4b62b0bdf964a520204f2ae3', '4ba5d2c9f964a520422439e3',
    '4baa99a3f964a52068783ae3', '4baf5d61f964a5207ffa3be3',
    '4c0e5b822466a5936de17821', '4c19d3cc4ff90f47168d1049',
    '4c91d1bb57e5b60c7acd631c', '4cd40a7da61c46881eb7b628',
    '4e68d3f41838d179b127207e', '4f2a8588e4b0114e618db8df',
    '5079cea8e4b0dad8227f29e3', '50edb02be4b0de6bf9c94534',
    '5173df84e4b0bb056be47496', '51dedf2d498e9dd2706d5b72',
    '5238643c11d2c1029cc5e106', '528b11db11d2330ae60ef113',
    '52f2130c498e7c57f7b0abab', '52fe522b11d256f8b35186c4']];
function marks_venues(clusters) {
	answers.clearLayers();
    var icon_color = ['black', 'red', 'blue', 'green', 'orange', 'purple',
        'darkpuple', 'cadetblue', 'darkred', 'darkgreen'];
    var gold = PRESETS[LAST_PRESET_NAME][dest];
    var best_dst = 1e20,
        best_radius = -1,
        nb_venues = 0;
    for (var k = 0; k < gold.length; k++) {
	    if (gold[k].metric !== LAST_USED_METRIC) {continue;}
        if (gold[k].dst < best_dst) {
            best_dst = gold[k].dst;
            best_radius = gold[k].geo.radius;
            nb_venues = gold[k].nb_venues;
        }
    }
    var gold = PRESETS[LAST_PRESET_NAME].gold[dest];
    if (gold) {
        for (var m = 0; m < gold.length; m++) {
			answers.addLayer(geojson_to_polygon(gold[m].geometry));
        }
    }
    msg = HTML(RESULT_FMT, {dst: best_dst.toFixed(3), nb_venues: nb_venues,
                                        radius: best_radius.toFixed(0)});
    $('#res').fill(msg);
    for (var j = 0; j < clusters.length; j++) {
        var marker = make_icon(icon_color[j]);
        for (var i = 0; i < clusters[j].length; i++) {
            var venue_id = clusters[j][i];
            var dot = L.marker(VENUES_LOC[venue_id], {clickable: false, icon: marker});
            answers.addLayer(dot);
        }
    }
}
function geojson_to_polygon(geo) {
    //TODO: use GeoJSON Layer
    //http://leafletjs.com/reference.html#geojson
	var coords = geo.coordinates[0], latlngs = [];
    for (var i = 0; i < coords.length-1; i++) {
	    latlngs.push([coords[i][1], coords[i][0]]);
    }
    return L.polygon(latlngs, {color: '#b22222'});
}
function draw_query_region(query) {
	drawnItems.clearLayers();
	var i = 0;
	if (origin === 'paris' || query.gold[origin].length === 1) {
		var tmp = query;
		if (origin !== 'paris') {
			tmp = query.gold[origin][0];
			tmp.nb_venues = tmp.properties.nb_venues;
			tmp.geo = tmp.geometry;
		}
		$('#orig-venues').fill(tmp.nb_venues + ' venues.');
		var poly = geojson_to_polygon(tmp.geo);
		drawnItems.addLayer(poly);
		left.fitBounds(poly.getBounds(), {maxZoom: 14});
	}
	else {
		for (i = 0; i<query.gold[origin].length; i++) {
			drawnItems.addLayer(geojson_to_polygon(query.gold[origin][i].geometry));
		}
	}
}
function draw_preset_query(name) {
    var query = PRESETS[name];
    draw_query_region(query);
    res = query[dest];
    answers.clearLayers();
    if (origin !== 'paris') {
        return false;
    }
    var metric = $('#presets').values().metric;
    var smallest_dst = 1e15;
    console.log(res);
    for (var i = 0; i < res.length; i++) {
	    var nb_venues = res[i].nb_venues;
        if (nb_venues === 0) {continue;}
        var dst = res[i].dst,
            center = res[i].geo.center,
            radius = res[i].geo.radius,
            r_metric = res[i].metric;
        if (r_metric === metric) {
            var circle = L.circle(center, radius, {color: '#2ecc40', fillOpacity: 0.05});
            answers.addLayer(circle);
            var dot = L.marker(center, {clickable: false, title: radius.toFixed(0),
                opacity: 0.7, riseOnHover: true, icon: smallIcon});
            answers.addLayer(dot);
            if (dst < smallest_dst) {
                msg = HTML(RESULT_FMT, {dst: dst.toFixed(3), nb_venues: res[i].nb_venues,
                                        radius: radius.toFixed(0)});
                $('#res').fill(msg);
                smallest_dst = dst;
            }
        }
    }
    // right.fitBounds(circle.getBounds(), {maxZoom: 14});
}
var presets = $('#presets');
var LAST_PRESET_NAME = null;
var LAST_USED_METRIC = null;
// if (origin !== 'paris') {presets.hide();}
presets.on('submit', function match_preset(e) {
    var form = presets.values();
    console.log(form);
    LAST_PRESET_NAME = form.neighborhood;
    LAST_USED_METRIC = form.metric;
    if (form.candidates === 'full') {
        draw_preset_query(form.neighborhood);
    }
    else {
        var query = PRESETS[form.neighborhood];
        draw_query_region(query);
        var geo = (origin === 'paris') ? query.geo : query.gold[origin][0].geometry;
        search_seed(form, geo);
    }
    return false;
});
function search_seed(input_values, query_geo) {
    var metric = input_values.metric,
        candidate = input_values.candidates,
        clustering = input_values.cluster;
    if (!query_geo.getBounds) {
        query_geo = geojson_to_polygon(query_geo);
    }
    left.fitBounds(query_geo.getBounds(), {maxZoom: 14});
    var json_geo = query_geo.toGeoJSON().geometry;
    $.request('post', $SCRIPT_ROOT+'/seed_region',
            {geo: JSON.stringify(json_geo), metric: metric, candidate: candidate,
                clustering: clustering})
    .then(function success(result){
        var answer = $.parseJSON(result).r;
        marks_venues(answer);
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
}
$(function() {
    $("#switch").onClick(function() {
        window.location.replace('/n/'+dest+'/'+origin);
    });
    window.setTimeout(function() {
    // marks_venues(TRIANGLE_VENUES);
    }, 500);
});
