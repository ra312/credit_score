#dummy-service       2020-04-16T12:51:55  http://b2237d27-7f81-4c49-850b-2415005e6d22.westeurope.azurecontainer.io/score
import requests
import json
scoring_uri = 'http://8e1891de-0ff3-4d73-969e-469fab494cb8.westeurope.azurecontainer.io/score'
headers = {'Content-Type':'application/json'}

test_data = json.dumps({'data': 
[[112123100107,1253,1,291.23277273859975,7,18.0,0.0]],
'method':'predict'
})

response = requests.post(scoring_uri, data=test_data, headers=headers)
print(response.status_code)
print(response.elapsed)
print(response)
