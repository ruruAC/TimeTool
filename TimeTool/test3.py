from urllib import request, parse

# Base URL being accessed
url = 'http://120.26.89.97:10000'

# Dictionary of query parameters (if any)
# parms = {
#    'data': [1],
#    'split': 1
# }

# Encode the query string
# querystring = parse.urlencode(parms)

# Make a GET request and read the response

from urllib import request, parse

# Base URL being accessed


# Dictionary of query parameters (if any)
parms = {
   'name1' : 'value1',
   'name2' : 'value2'
}

# Encode the query string
querystring = parse.urlencode(parms)

# Make a POST request and read the response


u = request.urlopen(url)

resp = u.read()
print(resp)