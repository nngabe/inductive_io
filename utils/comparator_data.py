import requests
import pandas as pd
import numpy as np
import os
import re
import time

# get credentials for Twitter API
bearer_token = os.environ.get('BEARER_TOKEN')
headers = {"Authorization": "Bearer {}".format(bearer_token)}
end_date = '2021-09-21T21:29:58Z'

def username_to_node(unames):
    """
        Pull tweets from by usernames.

        username_to_node(unames):
            args: usernames of twitter accounts
            returns: DataFrame with uid and other account attributes
    """
    df = {}
    for i,r in enumerate(unames):
        try:
            url = f'https://api.twitter.com/2/users/by?usernames={r}&user.fields=created_at,public_metrics&expansions=pinned_tweet_id&tweet.fields=author_id,created_at'
            response = requests.request("GET", url, headers=headers).json()
            df[r] = pd.DataFrame(response['data'])
            if i%100 == 0:
                print(f'hit({i}/{len(unames)}): {r}')
        except Exception as e:
            print(f'missed({i}/{len(unames)}):{r}')
            if 'errors' in response:
                continue
            elif 'title' in response:
                print('sleeping 16 minutes...')
                time.sleep(16*60)
            else:
                print(response)
            try:
                response = requests.request("GET", url, headers=headers).json()
                df[r] = pd.DataFrame(response['data'])
                print(f'hit({i}/{len(unames)}): {r}')
            except:
                print(f'missed({i}/{len(unames)}):{r}')
                
    res = pd.concat(df).reset_index().iloc[:,2:-1]
    keys = list(res.public_metrics[0].keys())
    for k in keys:
        res[k] = res.public_metrics.apply(lambda x: x[k])
    cols = ['username', 'id', 'created_at', 'followers_count', 'following_count',
       'tweet_count']
           
    return res.loc[:,cols]

def expand_urls(paths):
    "expand urls in *.csv and save *_exp.csv in same directory"
    for p in paths:
        df = pd.read_csv(p,index_col=0)
        short = pd.read_csv('list.txt',header=None).values.squeeze().tolist()
        df['short'] = df.domain.apply(lambda x: x in short)

        losu = df.url[df['short']].tolist()
        print(p)
        print(f'num. short urls: {len(losu)}')
        nomen = ''.join(p.split('_')[-3:-1]).split('/')[-1]
        print(nomen)
        exp = urlexpander.expand(losu, chunksize=1280, n_workers=64, cache_file=f'tmp{nomen}.json')

        fout = p[:-4] + '_exp' + p[-4:]
        print(fout,'\n')

        df['url_exp2'] = None
        df.url_exp2[df['short']] = exp
        df.to_csv(fout)

def union_urls(paths):
    "union urls and save *_union.csv in same directory"
    res = None

def grab_comparator_data(uids, cached_nodes=None):
    """
        Grab data by uid.

        grab_comparator_data(uids):
          args: uids - list of uids
          returns: None (writes tweets by account to ~/comp_data)
    """
    #end_date = '2021-09-21T21:29:58Z'
    for u in uids:
        url = f'https://api.twitter.com/2/users/{u}/tweets?tweet.fields=created_at,entities,public_metrics&max_results=5'
        print(f'starting uid {u}')
        try:
            response = requests.request("GET", url, headers=headers).json()
            res = pd.DataFrame(response['data'])
            r = res
        except Exception as e:
            print(e)
            print('1:')
            print(response)
            continue
        for j in range(50):
            oid = response['meta']['oldest_id']
            url = f'https://api.twitter.com/2/users/{u}/tweets?tweet.fields=created_at,entities,public_metrics&until_id={oid}&max_results=100'
            try:
                response = requests.request("GET", url, headers=headers).json()
                res = pd.DataFrame(response['data'])
                r = pd.concat([r,res])
                n = len(r)
                if j%15 ==0:
                    print(f'{u}: {n} tweets')
            except Exception as e:
                print(e)
                break
        print('Done.\n\n')
        path = f'../comp_data/tweet_data_{u}.csv'
        if os.path.exists(path):
            r.to_csv(f'../comp_data/tweet_data_{u}.csv', mode='a')
            time.sleep(1)
            print('(a) sleeping 1 seconds...')
        else:
            r.to_csv(f'../comp_data/tweet_data_{u}.csv')
            time.sleep(1)
            print('(w) sleeping 1 seconds...')

def preprocess_api_tweets(p):
    """ 
        Select relevant data from Twitter API requests and perform various 
        parsing and transformation steps before saving to a Pandas dataframe.
    """

    print(f'processing {p} ...')
    df = pd.read_csv(p,index_col=0)
    cols = ['screen_name', 'id', 'created_at', 'full_text', 'is_retweet',
           'retweeted', 'mentions', 'url_exp', 'url']
    df.url[~df.url.isna()] = df.url[~df.url.isna()].apply(ast.literal_eval)
    df['url_exp'] = df.urls_exp[~df.urls_exp.isna()].apply(ast.literal_eval)
    df = df.loc[:,cols]

    def url_tuple(x): 
        try:
            res = [[x1,x2] for x1,x2 in zip(x['url'],x['url_exp'])]
        except:
            res = ['','']
        return res
    df['url_pairs'] = df.apply(url_tuple, axis=1)

    df.url_pairs = df.url_pairs.apply(lambda x: [[None,None]] if len(x)==0 else x)

    def try_get(x,i):
        try:
            res = x[i]
        except:
            res = ''
        return res

    data2 = df.explode('url_pairs')
    data2.url_pairs = data2.url_pairs.apply(lambda x: [None,None] if len(x)==0 else x)
    data2.url = data2.url_pairs.apply(lambda x: try_get(x,0))
    data2.url_exp = data2.url_pairs.apply(lambda x: try_get(x,1))
    data2['domain'] = None
    data2.domain[~data2.url.isna()] = data2.url[~data2.url.isna()].apply(dom).to_list()
    data2['domain_exp'] = None
    data2.domain_exp[~data2.url.isna()]= data2.url_exp[~data2.url.isna()].apply(dom).to_list()
    out_path = p[:-4] + '_exp' + p[-4:]
    data2.to_csv(out_path)
    print(f'data written to {out_path} (Done)')
