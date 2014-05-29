var graph = create_graph('graph', 0.65, 0.4);
// plot(graph, [[.32, .85, 1.52, .73, -.51, -.25], [.23, .58, 1.25, .37, -.15, -.52]], ["hello", 'world']);
var left = create_map('mapl', LBBOX);
var right = create_map('mapr', RBBOX);
var MARKERS = [{}, {}];
var NAMES = [{}, {}];
var LOCS = [{}, {}];
var CATS = [{}, {}];
populate('right', display_venues);
populate('left', display_venues);
var CARD = '<a href="{{url}}" target="_blank">{{name}}</a>, {{cat}}<br>';
CARD += '<form onsubmit="match(\'{{_id}}\', {{nside}});return false;">';
CARD += '<button type="submit" autofocus>Match!</button></form>';
function display_venues(result, nside, map){
    var venues = $.parseJSON(result).r;
    _.each(venues, function add_venue(venue) {
        venue.nside = nside;
        var marker = L.marker(venue.loc, {title: venue.name, icon: smallIcon})
        .bindPopup(_.formatHtml(CARD, venue))
        .addTo(map);
    MARKERS[nside][venue._id] = marker;
    NAMES[nside][venue._id] = venue.name;
    LOCS[nside][venue._id] = L.latLng(venue.loc);
    CATS[nside][venue._id] = venue.cat;
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
            for (var j=0; j<KNN; j++) {
                table += _.formatHtml(cell_end, explanations[j][f]);
                gdata[j+1][feature_idx] = parseFloat(explanations[j][f].answer);
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
