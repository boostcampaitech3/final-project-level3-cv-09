from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

path = 'images/1'
search = 'bloody scene in movie'

# Google
google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': path})
#filters = dict(
    # size='large',
    # color='red',
    #license='commercial,modify',
    # date=((2017, 1, 1), (2022, 5, 30)))
google_crawler.crawl(keyword=search, filters=None, offset=0, max_num=3000,
                     min_size=(200,200), max_size=None, file_idx_offset=0)

# # bing
# bing_crawler = BingImageCrawler(downloader_threads=4,
#                                 storage={'root_dir': path})
# bing_crawler.crawl(keyword=search, filters=None, offset=0, max_num=1000)

# # baidu
# baidu_crawler = BaiduImageCrawler(storage={'root_dir': path})
# baidu_crawler.crawl(keyword=search, offset=0, max_num=1000,
#                     min_size=(200,200), max_size=None)