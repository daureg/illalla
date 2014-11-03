var ISIZE = 900;
var casper = require('casper').create({
    verbose: true,
    logLevel: "debug",
    viewportSize: {width: ISIZE, height: ISIZE}
});
var BASE_URL = 'http://0.0.0.0:5000/gold/TARGET/DISTRICT';
var NEI = ['16th', 'latin', 'marais', 'montmartre', 'official', 'pigalle',
	'triangle', 'weekend'];
var CASES = [];
var CITY = 'barcelona';
for (var i = 0, len = NEI.length; i < len; i++) {
	CASES.push({target: CITY, district: NEI[i]});
}
function case_to_name(full_case) {
	return full_case.target + '_' + full_case.district + '_gold.png';
}
function case_to_url(full_case) {
	var a = BASE_URL.replace('TARGET', full_case.target);
	return a.replace('DISTRICT', full_case.district);
}
casper.start("http://0.0.0.0:5000/");
casper.each(CASES, function(casper, tcase) {
    var fcase = JSON.parse(JSON.stringify(tcase));
    this.thenOpen(case_to_url(tcase), function() {
        this.wait(3000, function takeScreenshot() {
            this.capture('show_gold/'+case_to_name(tcase), {top: 0, left: 0, width: ISIZE, height: ISIZE});
        });
    });
});
casper.run();
