var casper = require('casper').create({
    verbose: true,
    logLevel: "debug",
    viewportSize: {width: 1905, height: 1200}
});
var BASE_URL = "http://0.0.0.0:5000/n/paris/",
    CITIES = ['sanfrancisco', 'barcelona'],
    NEIGHBORHOOD = ["triangle", "latin", "montmartre", "pigalle", "marais", "official"],
    METHODS = ['knn', 'jsd', 'emd'],
    CLUSTERING = ['dbscan', 'discrepancy'],
    CASES = [],
    tcase = null;
var debug = [
{url: BASE_URL+'barcelona', neighborhood: 'triangle', cluster: 'dbscan', metric: 'jsd', candidates: 'knn'},
{url: BASE_URL+'barcelona', neighborhood: 'latin', cluster: 'dbscan', metric: 'jsd', candidates: 'dst'},
{url: BASE_URL+'sanfrancisco', neighborhood: 'marais', cluster: 'discrepancy', metric: 'jsd', candidates: 'knn'},
    ];
for (var i = 0; i < CITIES.length; i++) {
    tcase = {url: BASE_URL + CITIES[i]};
    for (var j = 0; j < NEIGHBORHOOD.length; j++) {
        tcase.neighborhood = NEIGHBORHOOD[j];
        for (var k = 0; k < CLUSTERING.length; k++) {
            tcase.cluster = CLUSTERING[k];
            for (var l = 0; l < METHODS.length; l++) {
                tcase.metric = 'jsd';
                tcase.candidates = 'dst';
                if (METHODS[l] === 'emd') {tcase.metric = 'emd';}
                if (METHODS[l] === 'knn') {tcase.candidates = 'knn';}
                CASES.push(JSON.parse(JSON.stringify(tcase)));
            }
        }
    }
}

function case_to_name(full_case) {
    return full_case.neighborhood + '_' + full_case.url.split('/')[5] + '_' + full_case.candidates + '_' + full_case.metric + '_' + full_case.cluster;
}
casper.start(debug[0].url);
casper.each(CASES, function(casper, tcase) {
    var fcase = JSON.parse(JSON.stringify(tcase));
    delete fcase.url;
    this.thenOpen(tcase.url, function() {
        this.fill('form#presets', fcase, true);
        this.waitForSelectorTextChange('#log', function takeScreenshot() {
            this.wait(300, function() {
                this.capture('candidates/'+case_to_name(tcase)+'.png', {top: 30, left: 0, width: 1905, height: 1170});
            });
        }, function() {}, 90000);
    });
});
casper.run();
