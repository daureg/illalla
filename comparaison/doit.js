var casper = require('casper').create({
    verbose: true,
    logLevel: "debug",
    viewportSize: {width: 1900, height: 1200}
});
casper.start("http://0.0.0.0:5000/n/paris/sanfrancisco?triangle", function() { this.wait(2500, function() {
        this.capture('sf_triangle.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/sanfrancisco?latin", function() { this.wait(2500, function() {
        this.capture('sf_latin.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/sanfrancisco?marais", function() { this.wait(2500, function() {
        this.capture('sf_marais.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/sanfrancisco?pigalle", function() { this.wait(2500, function() {
        this.capture('sf_pigalle.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/sanfrancisco?official", function() { this.wait(2500, function() {
        this.capture('sf_official.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/sanfrancisco?montmartre", function() { this.wait(2500, function() {
        this.capture('sf_montmartre.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/sanfrancisco?weekend", function() { this.wait(2500, function() {
        this.capture('sf_weekend.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/sanfrancisco?16th", function() { this.wait(2500, function() {
        this.capture('sf_16th.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.start("http://0.0.0.0:5000/n/paris/barcelona?triangle", function() { this.wait(2500, function() {
        this.capture('bc_triangle.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/barcelona?latin", function() { this.wait(2500, function() {
        this.capture('bc_latin.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/barcelona?marais", function() { this.wait(2500, function() {
        this.capture('bc_marais.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/barcelona?pigalle", function() { this.wait(2500, function() {
        this.capture('bc_pigalle.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/barcelona?official", function() { this.wait(2500, function() {
        this.capture('bc_official.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/barcelona?montmartre", function() { this.wait(2500, function() {
        this.capture('bc_montmartre.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/barcelona?weekend", function() { this.wait(2500, function() {
        this.capture('bc_weekend.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.thenOpen("http://0.0.0.0:5000/n/paris/barcelona?16th", function() { this.wait(2500, function() {
        this.capture('bc_16th.png', {top: 0, left: 0, width: 1900, height: 1199}); }); });
casper.run();
