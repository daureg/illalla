var MINI = require('minified');
var _=MINI._, $=MINI.$, $$=MINI.$$, EE=MINI.EE, HTML=MINI.HTML;
L.Icon.Default.imagePath = '/static/images';

/* return the LatLng in the middle of an array of LatLng */
function barycenter(points) {
    var lat = 0, lng = 0, n = points.length;
    for (var i = n - 1; i >= 0; i--) {
        lat += points[i].lat;
        lng += points[i].lng;
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

function create_map(div_id, bbox) {
	var carto_layer = L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
	{attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'});
    var bbounds = compute_bound(bbox);
    var center = new L.LatLng(0.5*(bbox[0][0]+bbox[2][0]),
                              0.5*(bbox[0][1]+bbox[2][1]));
    var map = new L.Map(div_id, {zoom: 14, minZoom: 10, center: center,
                                 layers: [carto_layer], maxBounds: bbounds})
        .fitBounds(bbounds);
    L.polygon(bbox, {fill: false, weight: 3}).addTo(map);
    return map;
}
var left = create_map('mapl', LBBOX);
var right = create_map('mapr', RBBOX);
var MARKERS = [{}, {}];
var NAMES = [{}, {}];
var LOCS = [{}, {}];
populate('right');
populate('left');
function populate(side) {
    var origin = null, map = null, nside = null;
    if (side === 'left') {nside = 0; origin = true; map = left;}
    else {nside = 1; origin = false; map = right;}
    var card = '<a href="{{url}}" target="_blank">{{name}}</a>, {{cat}}<br>';
    card += '<form onsubmit="match();return false;">';
    card += '<input type="hidden" name="_id" value="{{_id}}">';
    card += '<input type="hidden" name="side" value="'+nside+'">';
    card += '<button type="submit" autofocus>Match!</button></form>';
    $.request('post', $SCRIPT_ROOT+'/populate', {origin: origin})
    .then(function success(result) {
        var venues = $.parseJSON(result).r;
        _.each(venues, function add_venue(venue) {
            var marker = L.marker(venue.loc, {title: venue.name})
                         .bindPopup(_.formatHtml(card, venue))
                         .addTo(map);
            MARKERS[nside][venue._id] = marker;
            NAMES[nside][venue._id] = venue.name;
            LOCS[nside][venue._id] = L.latLng(venue.loc);
        });
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
}
function match() {
    var request = $('form').values();
    request.side = parseInt(request.side);
    var other_side = (request.side + 1) % 2;
    var vids = [];
    var map = null;
    var cell = '<tr><td>{{query}}</td><td>{{feature}}</td>';
    cell += '<td><span style="color: {{color}};">{{percentage}}%</span></td>';
    cell += '<td>{{answer}}</td></tr>';
    $.request('post', $SCRIPT_ROOT+'/match', request)
    .then(function success(result) {
        var why = $.parseJSON(result).r;
        if (request.side === 0) { vids = [request._id, why._id]; map = right; }
        else { vids = [why._id, request._id];  map = left;}
        MARKERS[request.side][request._id].closePopup();
        MARKERS[other_side][why._id].openPopup();
        map.fitBounds(L.latLngBounds([LOCS[other_side][why._id]]), {maxZoom: 17});
        var table = '<table><thead><tr><td>'+NAMES[0][vids[0]]+'</td>';
        table += '<td>Feature</td><td>'+why.distance+'</td>';
        table += '<td>'+NAMES[1][vids[1]]+'</td></tr></thead>';
        table += '<tbody>';
        for (var i = 0; i < why.explanation.length; i++) {
            table += _.formatHtml(cell, why.explanation[i]);
        }
        table += '</tbody></table>';
        console.log(HTML(table));
        $('table').replace(HTML(table));
        console.log(why);
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
}
