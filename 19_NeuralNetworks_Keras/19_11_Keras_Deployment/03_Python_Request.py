import requests
import json

url = "http://127.0.0.1:5000/api/flower"

payload = json.dumps({
  "sepal_length": 5.4,
  "sepal_width": 3.3,
  "petal_length": 1.2,
  "petal_width": 0.15
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

# That should return 200 if correct
#print(response.status_code)

print(response.text)