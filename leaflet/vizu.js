// code for layer providers:
// http://leaflet-extras.github.io/leaflet-providers/preview/
function linerp(oa, ob, x, na, nb) {
	return na + (x - oa)*(nb -na)/(ob - oa);
}
function marker_from_photos(photos) {
	var count = photos.value.toString();
	var clamped = linerp(Math.log(5), Math.log(268), Math.log(count), .1, 1);
	L.marker([photos.lat, photos.lon],
			{title: count, riseOnHover: true, opacity: clamped})
		.bindPopup(count)
		.addTo(map);
}
var OpenStreetMap_HOT = L.tileLayer('http://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
	attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Tiles courtesy of <a href="http://hot.openstreetmap.org/" target="_blank">Humanitarian OpenStreetMap Team</a>'
});
var OpenStreetMap_Mapnik = L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
	attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'
});
var heatmapLayer = L.TileLayer.heatMap({radius: {value: 10}, opacity: 0.8});
// heatmapLayer.setData(helsinki.data);
var Stamen_Watercolor = new L.StamenTileLayer("watercolor");
var map = new L.Map('map', {
	center: new L.LatLng(60.194, 24.93),
	zoom: 12,
	layers: [
	// Stamen_Watercolor,
	OpenStreetMap_HOT,
	// heatmapLayer,
	]
});
// helsinki.data.forEach(marker_from_photos);
var city = L.polygon([
		[60.1463, 24.839],
		[60.242, 24.839],
		[60.242, 25.02],
		[60.1463, 25.02]
], {fill: false, weight: 3}).addTo(map);

var markers = L.markerClusterGroup();

var nb_points = helsinki_fs.length;
for (var i = 0; i < nb_points; i++) {
	var p = helsinki_fs[i];
	var title = p[2].toString();
	var marker = L.marker(new L.LatLng(p[1], p[0]), { title: title });
	marker.bindPopup(title);
	markers.addLayer(marker);
}

map.addLayer(markers);
