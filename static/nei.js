var left = create_map('mapl', LBBOX, {zoomAnimation: false});
var right = create_map('mapr', RBBOX, {zoomAnimation: false});
var map_right_toggle = $('#mapr').toggle({$$fade: 1}, {$$fade: 0}, 150);
populate('left', canvas_display);
populate('right', canvas_display);
var LEFT_CANVAS = null, LC_VISIBLE = false;
function canvas_display(result, nside, map) {
    var venues = $.parseJSON(result).r;
    var points = [];
    _.each(venues, function add_venue(venue) {
        var d = {"slat": venue.loc[0], "slon": venue.loc[1]};
        points.push(d);
    });
    var venue_dots = new MyLayer();
    venue_dots.setData(points);
    if (nside === 0) {
        LEFT_CANVAS = venue_dots;
        LC_VISIBLE = true;
    }
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
    var metric = $('#presets').values().metric;
    $.request('post', $SCRIPT_ROOT+'/match_neighborhood',
            {geo: JSON.stringify(geo), metric: metric})
    .then(function success(result){
        poll_until_done();
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
    if (!LC_VISIBLE) {
	    left.addLayer(LEFT_CANVAS);
	    LC_VISIBLE = true;
    }
});
var result_circle = null;
function poll_until_done() {
    $.request('get', $SCRIPT_ROOT+'/status', {})
    .then(function success(result){
        var answer = $.parseJSON(result).r;
        // console.log(answer);
        $('#status').set('@value', answer.progress);
        $('#res').fill('Smallest distance: ' + answer.res[0].toFixed(3));
        var radius = answer.res[1],
            center = answer.res[2];
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
function draw_preset_query(name) {
    var query = PRESETS[name];
    var coords = query.geo.coordinates[0], latlngs = [];
    var i = 0;
    for (i = 0; i < coords.length-1; i++) {
	    latlngs.push([coords[i][1], coords[i][0]]);
    }
    var poly = L.polygon(latlngs, {color: '#b22222'});
    drawnItems.clearLayers();
    drawnItems.addLayer(poly);
    left.fitBounds(poly.getBounds(), {maxZoom: 14});
    res = query[dest];
    answers.clearLayers();
    var metric = $('#presets').values().metric;
    var smallest_dst = 1e15;
    console.log(metric);
    console.log(query);
    for (i = 0; i < res.length-1; i++) {
        var dst = res[i].dst,
            center = res[i].geo.center,
            // radius = res[i].geo.radius,
            radius = res[i].radius,
            r_metric = res[i].metric;
        if (r_metric === metric) {
            circle = L.circle(center, radius, {color: '#2ecc40'});
            answers.addLayer(circle);
            if (dst < smallest_dst) {
                msg = 'Smallest distance of  ' + dst.toFixed(3);
                msg += ' for a radius of ' + radius.toFixed(0);
                $('#res').fill(msg);
                smallest_dst = dst;
            }
        }
    }
    // right.fitBounds(circle.getBounds(), {maxZoom: 14});
}
var presets = $('#presets');
if (origin !== 'paris') {presets.hide();}
presets.on('submit', function match_preset(e) {
    console.log(presets.values());
    draw_preset_query(presets.values().neighborhood);
    return false;
});
