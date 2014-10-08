var casper = require('casper').create({
    verbose: true,
    logLevel: "debug",
    viewportSize: {width: 1280, height: 720}
});
var METRIC="emd"
var BASE_URL = "http://0.0.0.0:5000/gold/TARGET/DISTRICT?from=SOURCE&m="+METRIC;
CASES = [
/*
	  */
         {source: "paris",
	district: "official",
	  target:  "barcelona"},
         {source: "paris",
	district: "marais",
	  target:  "washington"},
         {source: "newyork",
	district: "latin",
	  target:  "sanfrancisco"},
	  /*
         {source: "washington",
	district: "16th",
	  target:  "newyork"},
         {source: "barcelona",
	district: "montmartre",
	  target:  "rome"},
         {source: "paris",
	district: "official",
	  target:  "barcelona"},
         {source: "washington",
	district: "montmartre",
	  target:  "rome"},
	  */
	  ];
function case_to_name(full_case) {
	return full_case.source + '_' + full_case.target + '_' + full_case.district + '_'+METRIC+'.png';
}
function case_to_url(full_case) {
	var a = BASE_URL.replace('TARGET', full_case.target);
	a = a.replace('SOURCE', full_case.source);
	return a.replace('DISTRICT', full_case.district);
}
casper.start("http://0.0.0.0:5000/");
casper.each(CASES, function(casper, tcase) {
    var fcase = JSON.parse(JSON.stringify(tcase));
    this.thenOpen(case_to_url(tcase), function() {
        this.wait(1000, function takeScreenshot() {
            this.capture('best_metric/'+case_to_name(tcase), {top: 0, left: 0, width: 1280, height: 720});
        });
    });
});
casper.run();
