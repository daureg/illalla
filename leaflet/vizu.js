// code for layer providers:
// http://leaflet-extras.github.io/leaflet-providers/preview/
function linerp(oa, ob, x, na, nb) {
	return na + (x - oa)*(nb -na)/(ob - oa);
}
var myIcon = L.divIcon({className: 'photo'});
function marker_from_photos(photos) {
	var count = photos.value.toString();
	var clamped = linerp(Math.log(5), Math.log(268), Math.log(count), 0.1, 1);
	L.marker([photos.lat, photos.lon],
			{title: count, riseOnHover: true, opacity: clamped,
				icon: L.divIcon({className: 'photo', html: count})})
		// .bindPopup(count)
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
var center = new L.LatLng(60.194, 24.93);
var bbox = [[60.1463, 24.839], [60.242, 24.839], [60.242, 25.02], [60.1463, 25.02]];
function create_map(div_id, center, main_layer, bbox) {
	var map = new L.Map(div_id, {zoom: 12, center: center, layers: [main_layer]});
	L.polygon(bbox, {fill: false, weight: 3}).addTo(map);
	return map;
}
var mapc = create_map('mapc', center, OpenStreetMap_Mapnik, bbox);
var mapp = create_map('mapp', center, OpenStreetMap_HOT, bbox);
// helsinki.data.forEach(marker_from_photos);

function get_cluster(data) {
	var checkins = L.markerClusterGroup();
	var nb_points = data.length;
	for (var i = 0; i < nb_points; i++) {
		var p = data[i];
		var title = p[2].toString();
		var marker = L.marker(new L.LatLng(p[1], p[0]), { title: title });
		marker.bindPopup(title);
		checkins.addLayer(marker);
	}
	return checkins;
}

mapc.addLayer(get_cluster(helsinki_fs));
mapp.addLayer(get_cluster(helsinki_cluster));
