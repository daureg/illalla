/* From https://gist.github.com/2579599
 * implementation heavily influenced by http://bl.ocks.org/1166403
 */
'use strict';
function create_graph(id, w, h) {
    // define dimensions of graph
    var m = [64, 64, 64, 64]; // margins
    var w = parseInt(w*window.innerWidth) - m[1] - m[3]; // width
    var h = parseInt(h*window.innerHeight) - m[0] - m[2]; // height
    // create a tooltip to display name of each line
    var tooltip = d3.select("body").append("div")   
        .attr("class", "tooltip")
        .style("opacity", 0);
    // Add an SVG element with the desired dimensions and margin.
    var graph = d3.select('#'+id).append("svg:svg")
        .attr("width", w + m[1] + m[3])
        .attr("height", h + m[0] + m[2])
        .append("svg:g")
        .attr("transform", "translate(" + m[3] + "," + m[0] + ")");
    return {g: graph, w: w, h: h, tooltip: tooltip};
}
function clear_graph() {
    d3.selectAll('svg path').remove();
    d3.selectAll('.axis').remove();
    d3.selectAll('.legend').remove();
}
function plot(graph, data, names) {
    var colors = colorbrewer.Set1[8];
    var colors = [ '#ff4136', '#0074d9', '#2ecc40', '#f012be', '#ff851b', '#39cccc'];
    var places = [];
    for (var i=0; i<names.length; i++) {
        places.push({name: names[i], color: colors[i%colors.length]});
    }
    var labels = ['art', 'education', 'food', 'night', 'recreation', 'shop', 'professional', 'residence', 'transport'];
    var labels = ['1 to 5','5 to 9','9 to 13','13 to 17','17 to 21','21 to 1'];
    var h = graph.h,
        w = graph.w,
        tooltip = graph.tooltip,
        graph = graph.g;
    var scaleX = d3.scale.ordinal().domain(labels).range(d3.range(0, w+1, w/(labels.length-1)));
    
    // Y scale will fit values from 0-10 within pixels h-0 (Note the inverted domain for the y-scale: bigger is up!)
    var extents = [d3.min(data.map(function(e){return d3.min(e);})),
                   d3.max(data.map(function(e){return d3.max(e);}))];
    var scaleY = d3.scale.linear().domain(extents).range([h, 0]).nice();

    // create a line function that can convert data[] into x and y points
    var line = d3.svg.line()
        .x(function(d,i) { return scaleX(i); })
        .y(function(d) { return scaleY(d); })
        .interpolate('cardinal').tension(0.9);

    // create xAxis
    var xAxis = d3.svg.axis().scale(scaleX).tickSize(-h);
    // Add the x-axis.
    graph.append("svg:g")
        .attr("id", "xaxis")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + h + ")")
        .call(xAxis);

    // create left yAxis
    var yAxisLeft = d3.svg.axis().scale(scaleY).ticks(6).orient("left");
    // Add the y-axis to the left
    graph.append("svg:g")
        .attr("id", "yaxis")
        .attr("class", "y axis")
        .attr("transform", "translate(-15,0)")
        .call(yAxisLeft);

    // Add the line by appending an svg:path element with the data line we created above
    // do this AFTER the axes above so that the line is above the tick-lines
    for (var i=0; i<data.length; i++) {
        var c = colors[i%colors.length];
        graph.append("svg:path").attr("d", line(data[i]))
            .attr('stroke', c)
            .attr('class', "venue_line")
            .attr('opacity', 0.85)
            .attr("stroke-width", 2);
        graph.append("svg:path").attr("d", line(data[i]))
            .attr('id', 'shape_'+i)
            .attr('Xcolor', c.toString())
            .attr('Xname', names[i])
            .attr("stroke", "rgba(0,0,0,0.0)")
            .attr("stroke-width", 12)
            .on("mouseover", function() {      
                var info = d3.select(this);
                var color = info.attr('Xcolor'),
                    name = info.attr('Xname');
                tooltip.html(name)  
                .style("left", (d3.event.pageX) + "px")     
                .style('background-color', d3.rgb(color).darker())
                .style("top", (d3.event.pageY - 40) + "px")    
                .style("opacity", 0.8);    
            })                  
            .on("mouseout", function() {       
                   tooltip.style("opacity", 0);   
            });
    }
    var legend = graph.append('g')
        .attr('class', 'legend')
        .attr("height", places.length*20)
        .attr("width", 200)
        .attr('transform', 'translate(-150,0)');
    legend.selectAll('rect')
        .data(places)
        .enter()
        .append('rect')
        .attr('x', w - 20)
        .attr('y', function(d, i){return i *  20;})
        .attr('width', 10)
        .attr('height', 10)
        .style('fill', function(d) { return d.color; });

    legend.selectAll('text')
        .data(places)
        .enter()
        .append('text')
        .attr('x', w - 6)
        .attr('y', function(d, i){ return (i *  20) + 10;})
        .text(function(d){ return d.name; });

}
