"""
data_utils.py

Modulo per la preparazione dei dati per il progetto CAMI (Classificatore Automatico di Metafore per l’Italiano).

Contiene funzioni per:
- Caricare il dataset da CSV (UTF-8), rimuovere duplicati e testi vuoti
- Suddividere in training e test set (80/20) e salvare i CSV
- Calcolare statistiche esplorative: distribuzione etichette, lunghezza media frasi, top-N per colonne
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare_data(csv_path='data/CAMI_dataset_v2.csv', test_size=0.2, random_state=42):
    """
    Carica il dataset CSV, rimuove duplicati e testi vuoti, suddivide in train/test.
    Salva due file: train.csv e test.csv nella directory corrente.
    
    Args:
        csv_path (str): Percorso al file CAMI_dataset_v2.csv (colonne: testo, etichetta, argomento, veicolo).
        test_size (float): Percentuale del dataset destinata al set di test.
        random_state (int): Seed per la riproducibilità della suddivisione.
    
    Returns:
        train_df (pd.DataFrame): DataFrame di training.
        test_df (pd.DataFrame): DataFrame di test.
    """
    # Carica con encoding UTF-8
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Rimuovi duplicati basati sulla colonna 'testo'
    df.drop_duplicates(subset='testo', inplace=True)
    
    # Rimuovi righe con testo vuoto o NaN
    df = df[df['testo'].notna() & df['testo'].str.strip().astype(bool)]
    
    # Suddivisione stratificata basata su 'etichetta'
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['etichetta']
    )
    
    # Salva i file su disco
    train_df.to_csv('train.csv', index=False, encoding='utf-8')
    test_df.to_csv('test.csv', index=False, encoding='utf-8')
    
    return train_df, test_df

def get_label_distribution(df):
    """
    Calcola la distribuzione delle etichette (metafora vs non metafora) in un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contenente la colonna 'etichetta'.
    
    Returns:
        pd.Series: Serie con conteggi per ogni etichetta.
    """
    return df['etichetta'].value_counts()

def get_average_sentence_length(df):
    """
    Calcola la lunghezza media delle frasi in termini di numero di parole.
    
    Args:
        df (pd.DataFrame): DataFrame contenente la colonna 'testo'.
    
    Returns:
        float: Lunghezza media (numero medio di parole per frase).
    """
    # Conteggio parole dividendo sullo spazio
    lengths = df['testo'].apply(lambda x: len(str(x).split()))
    return lengths.mean()

def get_top_n_items(df, column, n=10):
    """
    Restituisce i top-n valori più frequenti in una colonna del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame di input.
        column (str): Nome della colonna (es. 'argomento' o 'veicolo').
        n (int): Numero di elementi da restituire.
    
    Returns:
        pd.Series: Serie con i top-n valori e le loro frequenze.
    """
    return df[column].value_counts().head(n)