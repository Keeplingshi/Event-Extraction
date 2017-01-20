# -*- coding: utf-8 -*-
'''
Created on 2017年1月20日

@author: chenbin

哈工大LTP工具
'''
import urllib.parse
import urllib.request
import urllib.error


def ltp_ppl(text):
    content=""
    uri_base = "http://api.ltp-cloud.com/analysis/?"
    api_key  = "o8l312I5hsGcLUKXLdMBCruskujudYlQCWfdlVNp"
    text     = urllib.parse.quote(text)
    format   = "plain"
    pattern  = "ws"

    url      = (uri_base
               + "api_key=" + api_key + "&"
               + "text="    + text    + "&"
               + "format="  + format  + "&"
               + "pattern=" + pattern)

    try:
        response = urllib.request.urlopen(url)
        content  = response.read().decode('utf-8')
    except urllib.error.URLError as e:
        print(e)
    
    return content