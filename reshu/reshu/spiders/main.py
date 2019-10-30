import scrapy

class MainScraper(scrapy.Spider):
    name = "main"
    start_urls = [
        "https://rus-ege.sdamgia.ru/"
    ]

    def parse(self, response):
        for el in response.css('a::attr(href)').re('/test.*'):
            yield response.follow(el, callback=self.parse_test)

    def parse_test(self, response):
        for problem in response.css('.prob_maindiv'):
            num = problem.css('.prob_nums::text').get()
            body = problem.css('.pbody')[0].css('p').getall()
            text = problem.css('.probtext').get()
            answer = None
            solution = problem.css('.solution').get()
            if not '27' in num:
                answer = problem.css('.answer span').re('.*Ответ: (.*)')[0]
            yield dict(num=num, body=body, text=text, answer=answer, solution=solution)
