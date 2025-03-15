import pandas as pd  # Importerer pandas biblioteket, som bruges til at arbejde med DataFrame og dataanalyse
from sklearn.datasets import fetch_california_housing  # Importerer fetch_california_housing fra sklearn, som henter California Housing datasættet

def load_california_housing():  # Definerer en funktion til at hente California Housing datasættet og returnere det som en DataFrame
    """Henter California Housing dataset og returnerer det som en DataFrame."""  # Funktionsdokumentation, der forklarer formålet med funktionen
    data = fetch_california_housing()  # Henter California Housing datasættet ved hjælp af fetch_california_housing metoden fra sklearn

    # Konverter data til en pandas DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)  # Konverterer de indlæste data til en pandas DataFrame og tildeler kolonnenavne fra 'feature_names'

    # Tilføj målværdien (huspriser)
    df["MedHouseVal"] = data.target  # Tilføjer en ny kolonne "MedHouseVal" til DataFrame, som indeholder målvariablen (huspriser) fra 'target'

    return df  # Returnerer den oprettede DataFrame, som nu indeholder både input data og målvariabel

if __name__ == "__main__":  # Kontrollerer, om scriptet køres som hovedprogram
    df = load_california_housing()  # Kalder funktionen for at hente California Housing datasættet som en DataFrame
    print(df.head())  # Udskriver de første 5 rækker af DataFrame for at vise et hurtigt udsnit af dataene
