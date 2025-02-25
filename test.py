import APIClient

def main():
    Client = APIClient.NHLAPIClient()

    data = Client.get_todays_games_info()

    print(data)

if __name__ == '__main__':
    main()