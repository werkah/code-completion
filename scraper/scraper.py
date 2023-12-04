import requests
import os

# NOTE: You need to set the GITHUB_TOKEN environment variable
token = os.environ.get("GITHUB_TOKEN")


def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, "w") as file:
            file.write("\n".join(response.json()["payload"]["blob"]["rawLines"]))
        print(f"Downloaded: {destination}")
    else:
        print(f"Failed to download: {url}")


def search_and_download(query, language="python"):
    api_url = "https://api.github.com/search/code"

    headers = {"authorization": f"Bearer {token}"}

    params = {
        "q": f"{query} extension:py language:{language}",
        "sort": "stars",
        "order": "desc",
    }

    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code == 200:
        results = response.json()["items"]
        for result in results:
            try:
                file_url = result["html_url"]
                file_name = result["name"]
                destination = os.path.join("downloaded_files", file_name)

                download_file(file_url, destination)
            except Exception as e:
                print(e)
    else:
        print(f"Failed to perform GitHub search. Status code: {response.status_code}")


if __name__ == "__main__":
    search_and_download("machine learning")
