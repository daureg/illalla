'use strict';
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
var graph = create_graph('graph', 0.65, 0.4);
// plot(graph, [[.32, .85, 1.52, .73, -.51, -.25], [.23, .58, 1.25, .37, -.15, -.52]], ["hello", 'world']);
var left = create_map('mapl', LBBOX);
var right = create_map('mapr', RBBOX);
var MARKERS = [{}, {}];
var NAMES = [{}, {}];
var LOCS = [{}, {}];
var CATS = [{}, {}];
populate('right');
populate('left');
function populate(side) {
    var origin = null, map = null, nside = null;
    if (side === 'left') {nside = 0; origin = true; map = left;}
    else {nside = 1; origin = false; map = right;}
    var card = '<a href="{{url}}" target="_blank">{{name}}</a>, {{cat}}<br>';
    card += '<form onsubmit="match(\'{{_id}}\', '+nside+');return false;">';
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
            CATS[nside][venue._id] = venue.cat;
        });
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
}
var SMAP = {
    'art surrounding': 0,
    'education surrounding': 1,
    'food surrounding': 2,
    'night surrounding': 3,
    'recreation surrounding': 4,
    'shop surrounding': 5,
    'professional surrounding': 6,
    'residence surrounding': 7,
    'transport surrounding': 8
};
var TMAP = {
'activity at 1--5' : 0,
'activity at 5--9' : 1,
'activity at 9--13' : 2,
'activity at 13--17' : 3,
'activity at 17--21' : 4,
'activity at 21--1' : 5
};
var FMAP = TMAP;
function match(_id, side) {
    var request = {side: side, _id: _id};
    var other_side = (request.side + 1) % 2, query_side = request.side;
    var map = null;
    var cell_begin = '<tr><td>{{val}}</td><td>{{feature}}</td>';
    var cell_end = '<td class="value">{{answer}}</td>';
    cell_end += '<td><span style="color: {{color}};">{{percentage}}%</span></td>';
    var gnames = [NAMES[query_side][request._id]];
    var gdata = [[], ];
    function venue_name(side, id_) {
        var res = '<a href="https://foursquare.com/v/'+id_+'" target="_blank">';
        return res + NAMES[side][id_] + '</a> ('+CATS[side][id_]+')';
    }
    $.request('post', $SCRIPT_ROOT+'/match', request)
    .then(function success(result) {
        var why = $.parseJSON(result).r;
        var query = why.query, distances = why.distances,
            answers_id = why.answers_id, explanations = why.explanations,
            map = (request.side === 0) ? right : left;
        MARKERS[request.side][request._id].closePopup();
        MARKERS[other_side][answers_id[0]].openPopup();
        map.fitBounds(L.latLngBounds([LOCS[other_side][answers_id[0]]]), {maxZoom: 17});
        var table = '<table><thead><tr><td>';
        table += venue_name(query_side, request._id)+' ('+why.among+')</td>';
        table += '<td>Feature</td>';
        for (var i=0; i<KNN; i++) {
            table += '<td class="value">'+distances[i]+'</td>';
            table += '<td>'+venue_name(other_side, answers_id[i])+'</td>';
            gnames.push(NAMES[other_side][answers_id[i]]);
            gdata.push([]);
        }
        table += '</tr></thead><tbody>';
        var feature_idx = 0;
        for (var f = 0; f < explanations[0].length; f++) {
            table += _.formatHtml(cell_begin, query[f]);
            feature_idx = FMAP[query[f].feature];
            gdata[0][feature_idx] = parseFloat(query[f].val);
            for (var i=0; i<KNN; i++) {
                table += _.formatHtml(cell_end, explanations[i][f]);
                gdata[i+1][feature_idx] = parseFloat(explanations[i][f].answer);
            }
            table += '</tr>';
        }
        table += '</tbody></table>';
        $('table').replace(HTML(table));
        clear_graph();
        plot(graph, gdata, gnames);
    })
    .error(function(status, statusText, responseText) {
        console.log(status, statusText, responseText);
    });
}
