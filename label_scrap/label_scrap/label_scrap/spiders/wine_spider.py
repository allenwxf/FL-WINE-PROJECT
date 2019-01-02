import scrapy, requests, os

class WineSpider(scrapy.Spider):
    name = "wine"

    def start_requests(self):
        # urls = [
        #     "https://www.wine.com/list/wine/7155",
        #     "https://www.wine.com/list/wine/7155/2"
        # ]
        # for url in urls:
        baseurl = "https://www.wine.com/list/wine/7155/"
        for page in range(15000, 16000):
            yield scrapy.Request(url=baseurl+str(page), callback=self.parse)
    # start_urls = [
    #         "https://www.wine.com/list/wine/7155",
    #         "https://www.wine.com/list/wine/7155/2"
    #     ]

    def parse(self, response):
        # 图片保存地址
        img_save_dir = self.img_save_dir

        page = response.url.split("/")[-1]
        # filename = 'tmp/winelist-%s.html' % page
        # with open(filename, "wb") as f:
        #     f.write(response.body)
        # self.log("Saved file %s" % filename)

        # 详细链接
        detail_links = response.css("div.prodItemImage > .prodItemImage_link::attr(href)").extract()
        # 酒标图片
        labels = response.css("div.prodItemImage > .prodItemImage_link > .prodItemImage_image img::attr(src)").extract()
        # 酒名（包含年份）
        winenames = response.css("div.prodItemInfo > a.prodItemInfo_link > span.prodItemInfo_name::text").extract()
        # 品种
        varieties = response.css("div.prodItemInfo > div.prodItemInfo_body > div.prodItemInfo_origin > "
                     "span.prodItemInfo_varietal::text").extract()
        # 产地
        origins = response.css("div.prodItemInfo > div.prodItemInfo_body > div.prodItemInfo_origin > "
                     "span.prodItemInfo_originText::text").extract()

        # 专业评分
        ratings = []
        ratings_doms = response.css("div.prodItemInfo > div.prodItemInfo_body > ul.wineRatings_list")
        for ratings_dom in ratings_doms:
            this_ratings = ratings_dom.css("li::attr(title)").extract()
            ratings.append(this_ratings)

        # 平均星级
        avg_ratings = response.css("div.prodItemInfo > div.prodItemInfo_body > div.prodItemInfo_rating > div.averageRating > "
                     "span.averageRating_average::text").extract()

        # 星级评价人数
        rating_peoples = response.css("div.prodItemInfo > div.prodItemInfo_body > div.prodItemInfo_rating > "
                                      "div.averageRating > span.averageRating_count > "
                                      "span.averageRating_number::text").extract()

        # 价格
        prices = []
        price_doms = response.css("div.prodItemInfo > div.productPrice > div.productPrice_price-reg")
        for price_dom in price_doms:
            this_price_whole = price_dom.css("span.productPrice_price-regWhole::text").extract_first()
            this_price_fractional = price_dom.css("span.productPrice_price-regFractional::text").extract_first()
            if this_price_whole is None:
                this_price_whole = 0
            else:
                this_price_whole = this_price_whole.replace(",", "")
            if this_price_fractional is None:
                this_price_fractional = 0
            else:
                this_price_fractional = this_price_fractional.replace(",", "")
            this_price = int(this_price_whole) + int(this_price_fractional) * 0.01
            prices.append(this_price)

        if len(prices) != len(detail_links) :
            prices = []
            for item in detail_links:
                prices.append(0.0)

        wine_datas = zip(detail_links, labels, winenames, varieties, origins, ratings, avg_ratings, rating_peoples, prices)

        for wine_data in wine_datas:
            label_name = wine_data[1].replace("/", "_")

            if not os.path.exists(img_save_dir + "/" + label_name):
                ir = requests.get(response.urljoin(wine_data[1]))
                open(img_save_dir + "/" + label_name, "wb").write(ir.content)

            yield {
                "page": page,
                "detail_links": response.urljoin(wine_data[0]),
                "labels": response.urljoin(wine_data[1]),
                "label_name":label_name,
                "winenames": wine_data[2],
                "varieties": wine_data[3],
                "origins": wine_data[4],
                "ratings": wine_data[5],
                "avg_ratings": wine_data[6],
                "rating_peoples": wine_data[7],
                "prices": wine_data[8]

            }

