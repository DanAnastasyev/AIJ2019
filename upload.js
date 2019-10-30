console.log('Start');
var webPage = require('webpage');
var page = webPage.create();
var system = require('system');
var args = system.args;

page.open('https://contest.ai-journey.ru/', function(status) {
    console.log('Status: ' + status);
    console.log(page.title);
    console.log(phantom.addCookie({'name': 'sessionid', 'value': args[1], 'domain': 'contest.ai-journey.ru', 'path': '/'}));
    page.open('https://contest.ai-journey.ru/ru/team', function(status) {
        console.log('Status: ' + status);
        console.log(page.title);
        page.evaluate(function() {
            $("[data-target='#submitModal']").click();
        });
        console.log(page.uploadFile('#inputFile2', 'solution.zip'));
        setInterval(function () {
            console.log('Waiting ' + page.evaluate(function () {
                return $("#submit_progress").text();
            }));
            var good = page.evaluate(function() {
                return $("#subm_comp_btn").hasClass("join-button");
            });
            if (good) {
                page.evaluate(function(comment) {
                    $("#submit_comment_textarea").val(comment)
                    $("#subm_comp_btn").click();
                }, args[2]);
                setTimeout(function () {
                    page.render('res1.png');
                    phantom.exit();
                }, 1000);
            }
        }, 10000);
    });
 });
