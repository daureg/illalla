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

var filename = urlParams.file || 'EU.tsv';
if (filename.substr(filename.length - 4) !== '.tsv') {filename += '.tsv';}
var ALL_CITIES = filename !== 'pbh.tsv';
var city_state = -1,
    cat_state = -1;
var margin = {top: 5, right: 35, bottom: 20, left: 28},
    // width = 900 - margin.left - margin.right,
    // height = 500 - margin.top - margin.bottom;
    width = window.innerWidth - margin.left - margin.right,
    height = window.innerHeight - 27 - margin.top - margin.bottom;

var x = d3.scale.linear().range([0, width]);
var y = d3.scale.linear().range([height, 0]);

var color = d3.scale.category10();
var xAxis = d3.svg.axis().scale(x).orient("bottom");

var yAxis = d3.svg.axis().scale(y).orient("left");

var svg = d3.select("body").append("svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)
.append("g")
.attr("transform", "translate(" + margin.left + "," + margin.top + ")");


CATS = ['art', 'education', 'food', 'night', 'recreation', 'shop',
     'professional', 'residence', 'transport'];
var color = d3.scale.ordinal().domain(d3.range(CATS.length))
    .range(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#888888']);
CITIES = [];
CITIES_COUNT = {};
CATS_COUNT = [];

d3.tsv(filename, function(error, data) {
    data.forEach(function(d) {
        d.posx = +d.posx;
        d.posy = +d.posy;
        // d.info = '<a href="https://foursquare.com/v/'+d.id+'">'+d.name+'</a>';
        if (d.city in CITIES_COUNT) {CITIES_COUNT[d.city] += 1;}
        else {CITIES_COUNT[d.city] = 1; CITIES.push(d.city);}
        if (d.cat in CATS_COUNT) {CATS_COUNT[d.cat] += 1;}
        else {CATS_COUNT[d.cat] = 1;}
    });

    x.domain(d3.extent(data, function(d) { return d.posx; })).nice();
    y.domain(d3.extent(data, function(d) { return d.posy; })).nice();
    var zoom = d3.behavior.zoom().x(x).y(y).scaleExtent([1, 10]).on("zoom", zoomed);
    svg.call(zoom);

    function ztransform(d) {
        var rot = (!ALL_CITIES && city_to_shape(d.city) === 'diamond') ? "rotate(-45,"+ x(d.posx) + "," + y(d.posy) + ") ": '';
        return rot +"translate(" + x(d.posx) + "," + y(d.posy) + ")";
    }

    var city_to_shape = d3.scale.ordinal().domain(CITIES).range(['circle', 'diamond', 'square']);
    svg.append("rect")
    .attr("width", width)
    .attr("height", height);

    var x_ax = svg.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

    var y_ax = svg.append("g")
    .attr("class", "y axis")
    .call(yAxis);

    var shape = (ALL_CITIES) ? 'circle' : 'rect';
    var dots = svg.selectAll(".dot")
    .data(data)
    .enter().append(shape)
    .attr("class", function(d) { return city_to_shape(d.city)+' dot '+ CATS[d.cat] + ' '+ d.city;})
    .attr("width", 8)
    .attr("height", 8)
    .attr('rx', function(d) { return (!ALL_CITIES && city_to_shape(d.city) === 'circle') ?  '4px' : '0';})
    .attr('ry', function(d) { return (!ALL_CITIES && city_to_shape(d.city) === 'circle') ?  '4px' : '0';})
    .attr('r', ALL_CITIES ? 4 : 0)
    .attr("transform", ztransform)
    .style("fill", function(d) { return color(d.cat); });
    function zoomed() {
        x_ax.call(xAxis);
        y_ax.call(yAxis);
        dots.attr("transform", ztransform);
    }
    d3.select('#city').on('click', function() {city_legend_click(-1);});
    d3.select('#cat').on('click', function() {cat_legend_click(-1);});
    function city_legend_click(d) {
        var city = d;
        d = CITIES.indexOf(d);
        console.log('city :', d);
        if (city_state === d) {d = -1;}
        city_state = d;
        dots.style('opacity', 0.1).filter(should_be_visible).style('opacity', 1);
        document.getElementById('city').innerHTML = (d === -1) ? '' : CITIES[d]+': '+CITIES_COUNT[city];
    }
    function cat_legend_click(d) {
        if (cat_state === d) {d = -1;}
        cat_state = d;
        dots.style('opacity', 0.1).filter(should_be_visible).style('opacity', 1);
        document.getElementById('cat').innerHTML = (d === -1) ? '' : CATS[d]+': '+CATS_COUNT[d];
    }
    function should_be_visible(d) {
        if (city_state > -1 && CITIES.indexOf(d.city) != city_state) {return false;}
        if (cat_state > -1 && d.cat != cat_state) {return false;}
        return true;
    }
    var legend = svg.selectAll(".legend")
        .data(color.domain())
        .enter().append("g")
        .attr("class", "legend")
        .attr("transform", function(d, i) { return "translate(20," + i * 20 + ")"; });

    legend.append("rect")
        .attr("x", width - 18)
        .attr("width", 18)
        .attr("height", 18)
        .on('click', cat_legend_click)
        .style("fill", color);

    legend.append("text")
        .attr("x", width - 24)
        .attr("y", 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text(function(d) {return CATS[d]; });

    if (!ALL_CITIES) {
        var clegend = svg.selectAll(".clegend")
            .data(city_to_shape.domain())
            .enter().append("g")
            .attr("class", "legend")
            .attr("transform", function(d, i) { return "translate(0," + i * 25 + ")"; });

        clegend.append("rect")
            .attr("x", -20)
            .attr("width", 16)
            .attr("height", 16)
            .attr('rx', function(d) { return  (city_to_shape(d) === 'circle') ?  '9px' : '0';})
            .attr('ry', function(d) { return  (city_to_shape(d) === 'circle') ?  '9px' : '0';})
            .attr('transform', function(d, i) {
                return (!ALL_CITIES && city_to_shape(d) === 'diamond') ? "rotate(-45,"+ -8 + "," + 8 + ")": 'none';})
            .on('click', city_legend_click)
            .style("fill", '#222');

        clegend.append("text")
            .attr("x", 2)
            .attr("y", 9)
            .attr("dy", ".35em")
            .style("text-anchor", "begin")
            .text(function(d) {console.log(d); return d; });
    }
    save_svg_code();
});
function save_svg_code() {
	var MINI = require('minified');
	var _=MINI._, $=MINI.$;
	var svg = document.getElementsByTagName("svg")[0];
	var svg_xml = (new XMLSerializer).serializeToString(svg);
	svg_xml = vkbeautify.xml(svg_xml);
	$.request('post', 'http://localhost:8087/out', {xml: svg_xml})
	.then(function success(result){
		console.log(result);
	})
	.error(function(status, statusText, responseText) {
		console.log(status, statusText, responseText);
	});
	// document.getElementById("svg_code").text(svg_xml);
}
