from urllib import request, parse
import pandas as pd
def runlstm(df_orig,split):
   # Base URL being accessed
   url = 'http://localhost:8000/items/next'

   # # Dictionary of query parameters (if any)
   # parms = {
   #    'name1' : 'value1',
   #    'name2' : 'value2'
   # }



   # # Encode the query string
   # querystring = parse.urlencode(parms)
   print("___kkkk__")
   #print(parms['data'])


   #querystring = parse.urlencode(parms)



   # Make a GET request and read the response
   #u = request.urlopen(url + '?' + querystring)
   for col in df_orig:
      df_orig[col]=pd.to_numeric(df_orig[col])
   parms = {
      'data': df_orig,
      'split': split
   }
   querystring = parse.urlencode(parms)
   #df_orig.columns=['values']
   #print(df_orig)
   #print(url,"?",'data=',df_orig,"&",'split=',split)
   from downstream import writecsv
   #kk="http://120.26.89.97:8501/downloads/"+'out.csv'
   kk=writecsv(df_orig,'out.csv')
   print(kk)
   #df_orig.to_csv(kk, encoding='gb18030', index=True)

   print(url+"?"+'datapath='+str(kk)+'\out.csv'+"&"+'split='+str(split))
   u = request.Request(url+"?"+'datapath='+str(kk)+'\out.csv'+"&"+'split='+str(split))

   #u = request.Request(url, querystring.encode('ascii'))
   #print(u)
   resp = request.urlopen(u).read()

   #resp = u.read()

   return resp
