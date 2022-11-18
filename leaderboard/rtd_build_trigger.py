import requests

def build(args):
    version = args.version
    version = 'leaderboard-and-unitest-update'
    URL = f'https://readthedocs.org/api/v3/projects/pyhealth/versions/{version}/builds/'
    with open(args.rtd_token) as f:
        TOKEN = f.readlines()[0]
    HEADERS = {'Authorization': f'token {TOKEN}'}
    response = requests.post(URL, headers=HEADERS)
    print(response.json())


def construct_args():
    parser.add_argument("--rtd_token", type=str, default='./rtd_token.txt')
    parser.add_argument("--version", type=str, default='lastest')


def main():
    args = construct_args()
    build(args)


if __name__ == '__main__':
    main()

