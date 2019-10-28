console.log('Start');
var webPage = require('webpage');
var page = webPage.create();
var system = require('system');
var args = system.args;

console.log(phantom.addCookie({'name': 'sessionid', 'value': args[1], 'domain': 'contest.ai-journey.ru', 'path': '/'}));
page.open('https://contest.ai-journey.ru/ru/team', function(status) {
    console.log('Status: ' + status);
    console.log(page.evaluate(function() {
        return $("#submit-table").html();
    }));
    phantom.exit();
});
