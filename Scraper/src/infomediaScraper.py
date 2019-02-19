import requests

stdData = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Origin': 'https://apps.infomedia.dk',
        'X-Requested-With': 'XMLHttpRequest',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
        'Content-Type': 'application/json; charset=UTF-8',
        'Referer': 'https://apps.infomedia.dk/mediearkiv',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9,da-DK;q=0.8,da;q=0.7,sv;q=0.6',
        'Cookie': 'gsScrollPos-387=0; Insight.Web.Ms.SessionId=wlglnuxx3o0nqjtxs5vtztnz'
    }

stdDataIql = stdData
stdDataIql['iqlString'] = '\'martin henriksen\' AND (integ* OR indva* OR asyl*)'
stdDataIql['Accept'] = '*/*'
stdDataIql['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
stdDataIql['Cookie'] = 'gsScrollPos-387=0; Insight.Web.Ms.SessionId=wlglnuxx3o0nqjtxs5vtztnz'

searchDataTest = stdData
searchDataTest['iqlString'] = '\'martin henriksen\' AND (integ* OR indva* OR asyl*)'


def crawlInfomedia():
    session = requests.Session()
    session.get('https://login.infomedia.dk/')

    # Login
    session.get('https://login.infomedia.dk/Login/AutologinRedirect?appName=InsightLogin')

    # Navigate to Expert Search + get and set user settings
    testA = session.get('https://apps.infomedia.dk/mediearkiv/settings/setusersettings', data=stdData)
    testB = session.get('https://apps.infomedia.dk/mediearkiv/search/getiqlfrombasicfilter', data=stdData)
    testC = session.post('https://apps.infomedia.dk/mediearkiv/settings/getusersettings', data=stdData)
    print('testA', testA.status_code)
    print('testB', testB.status_code)
    print('testC', testC.status_code)
    # Status: Correctly navigated to Expert Search at this point
    # testX = session.get('https://apps.infomedia.dk/mediearkiv')
    # print(testX.text)

    # Test IQL Syntax
    testD = session.post('https://apps.infomedia.dk/mediearkiv/search/iqlsyntaxcheck', data=stdDataIql)
    print('testD', testD.status_code, '\n', testD.text)

    # Perform search
    testE = session.post('https://apps.infomedia.dk/mediearkiv/search/iqlsearch', data=stdData)
    print('testE', testE.status_code, '\n', testE.text)


crawlInfomedia()
