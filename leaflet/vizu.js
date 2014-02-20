// code for layer providers:
// http://leaflet-extras.github.io/leaflet-providers/preview/
var OpenStreetMap_HOT = L.tileLayer('http://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
	attribution: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Tiles courtesy of <a href="http://hot.openstreetmap.org/" target="_blank">Humanitarian OpenStreetMap Team</a>'
});
var heatmapLayer = L.TileLayer.heatMap({radius: 20, opacity: 0.6});
heatmapLayer.addData(helsinki);
// var map = L.map('map').setView([], 13);
// var Stamen_Watercolor = new L.StamenTileLayer("watercolor");
// map.addLayer(OpenStreetMap_HOT);
var overlayMaps = {'Heatmap': heatmapLayer};
var controls = L.control.layers(null, overlayMaps, {collapsed: false});
var map = new L.Map('map', {
	center: new L.LatLng(60.173324, 24.939927),
	zoom: 13,
	layers: [OpenStreetMap_HOT, heatmapLayer]
});
controls.addTo(map);
