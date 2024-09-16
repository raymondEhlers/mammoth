from __future__ import annotations

import json

import requests


def get_authors(inspire_id: int, max_authors: int = 10) -> list[str]:
    # Construct the API URL
    url = f"https://inspirehep.net/api/literature/{inspire_id}"

    # Send a GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = json.loads(response.text)

        # Extract the authors from the metadata
        authors = data["metadata"]["authors"]

        # Return the first max_authors authors
        return authors[:max_authors]

    print(f"Error: Unable to fetch data. Status code: {response.status_code}")
    return []


def retrieve_authors_list() -> None:
    inspire_ids = [
        2818238,
        2637686,
        2662562,
        1762358,
        1706143,
        1435094,
        2062342,
        2178285,
        2149709,
        # Proceedings. Skip
        # 2744879,
        # 2708671,
        # 2513630,
        # 2136843,
        # 1864763,
        # 1819261,
    ]
    max_authors = 2

    for inspire_id in inspire_ids:
        print(f"Checking on id {inspire_id}")

        # Retrieve authors and print
        authors = get_authors(inspire_id, max_authors=max_authors)
        if authors:
            authors = [author["full_name"] for author in authors]
            print(f"{inspire_id}: {' and '.join(authors)} and others")
        else:
            print("No authors found.")


if __name__ == "__main__":
    retrieve_authors_list()
