import requests

URL = 'https://readthedocs.org/api/v3/projects/pyhealth/versions/leaderboard-and-unitest-update/builds/'
with open('./rtd_token.txt') as f:
    TOKEN = f.readlines()[0]
HEADERS = {'Authorization': f'token {TOKEN}'}
response = requests.post(URL, headers=HEADERS)
print(response.json())