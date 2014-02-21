// code for layer providers:
// http://leaflet-extras.github.io/leaflet-providers/preview/
var OpenStreetMap_HOT = L.tileLayer('http://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
	attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Tiles courtesy of <a href="http://hot.openstreetmap.org/" target="_blank">Humanitarian OpenStreetMap Team</a>'
});
var heatmapLayer = L.TileLayer.heatMap({radius: {value: 20}, opacity: 0.7});
heatmapLayer.setData(helsinki.data);
// var Stamen_Watercolor = new L.StamenTileLayer("watercolor");
var map = new L.Map('map', {
	center: new L.LatLng(60.1733, 24.934),
	zoom: 14,
	layers: [OpenStreetMap_HOT,
	heatmapLayer,
	]
});
