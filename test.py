import requests

url = 'http://localhost:5050/'

parameters = {
    'user_message': 'choice no 2',
            }

response = requests.get(url, params=parameters)

if response.ok:
    print(response.text)
else:
    print(f"Error: {response.status_code}")
