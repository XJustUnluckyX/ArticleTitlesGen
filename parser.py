import requests
from bs4 import BeautifulSoup


def extract_data(url_param):
    extracted_abstracts = []

    response = requests.get(url_param)

    if response.status_code == 200:
        # Parsing della pagina
        soup = BeautifulSoup(response.text, 'html.parser')

        # Individuazione di tutti gli abstract
        abstract_elements = soup.find_all('div', class_='abstract')

        # Estrazione del testo
        extracted_abstracts = [abstract.text.strip() for abstract in abstract_elements]

    else:
        print("Errore: La richiesta HTTP non è andata a buon fine.")

    return extracted_abstracts


urls = []
abstracts = extract_data(urls)

if abstracts:
    print("Abstracts acquisiti con successo:")
    for idx, abstract in enumerate(abstracts, start=1):
        print(f"Abstract {idx}: {abstract}")
else:
    print("Errore durante l'acquisizione degli abstract.")
