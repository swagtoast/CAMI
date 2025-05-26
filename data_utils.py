import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Carica il dataset da un file CSV, effettua pulizia base e lo suddivide
    in training e test set.

    Args:
        csv_path (str): Path del file CSV.
        test_size (float): Proporzione del dataset da allocare al test set.
        random_state (int): Seed per la riproducibilità della suddivisione.

    Returns:
        tuple: (train_df, test_df) DataFrame di Pandas.
               Restituisce (None, None) se il caricamento fallisce o mancano colonne essenziali.
    """
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        logging.info(f"Dataset caricato da {csv_path} con {len(df)} righe.")
    except FileNotFoundError:
        logging.error(f"File non trovato: {csv_path}")
        return None, None
    except Exception as e:
        logging.error(f"Errore durante il caricamento del CSV {csv_path}: {e}")
        return None, None

    # Colonne richieste: 'testo' e 'etichetta'. 'argomento' e 'veicolo' sono opzionali ma utili.
    required_cols = ['testo', 'etichetta']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Il CSV deve contenere le colonne: {', '.join(required_cols)}")
        return None, None

    # Pulizia
    df.dropna(subset=['testo'], inplace=True) # Rimuovi righe con testo vuoto
    df.drop_duplicates(subset=['testo'], inplace=True) # Rimuovi duplicati basati sul testo
    df['testo'] = df['testo'].astype(str).str.strip() # Assicura che 'testo' sia stringa e senza spazi extra
    df = df[df['testo'] != ""] # Rimuovi testi che sono diventati vuoti dopo strip

    # Assicurati che 'etichetta' sia numerica (0 o 1)
    # Se 'etichetta' fosse testuale (es. 'metafora', 'non_metafora'), dovresti mapparla qui.
    # Esempio: df['etichetta'] = df['etichetta'].map({'metafora': 1, 'non_metafora': 0})
    # Per questo esempio, assumiamo che 'etichetta' sia già 0 o 1.
    if not pd.api.types.is_numeric_dtype(df['etichetta']):
        logging.warning("La colonna 'etichetta' non è numerica. Tentativo di conversione.")
        # Esempio di mappatura se le etichette fossero stringhe
        label_map = {'metafora': 1, 'non_metafora': 0, 'Metafora': 1, 'Non-Metafora': 0}
        if df['etichetta'].dtype == 'object' and any(item in label_map for item in df['etichetta'].unique()):
             df['etichetta'] = df['etichetta'].map(label_map)
        try:
            df['etichetta'] = pd.to_numeric(df['etichetta'])
        except ValueError:
            logging.error("Impossibile convertire 'etichetta' in numerico. Controlla i valori.")
            return None, None
    
    df.dropna(subset=['etichetta'], inplace=True) # Rimuovi righe dove l'etichetta non è valida dopo la conversione
    df['etichetta'] = df['etichetta'].astype(int)


    if len(df) < 2:
        logging.error("Dataset troppo piccolo dopo la pulizia per essere suddiviso.")
        return None, None
        
    if len(df['etichetta'].unique()) < 2:
        logging.warning("Attenzione: il dataset ha una sola classe dopo la pulizia. La stratificazione potrebbe fallire o non essere significativa.")
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        try:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['etichetta'])
        except ValueError as e:
            logging.warning(f"Errore durante la stratificazione (es. una classe ha pochi campioni): {e}. Suddivisione non stratificata.")
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    logging.info(f"Dataset suddiviso: {len(train_df)} righe per training, {len(test_df)} righe per test.")
    return train_df, test_df

# Funzioni di utilità per Analisi Esplorativa (EDA)
def get_label_distribution(df: pd.DataFrame):
    """Calcola la distribuzione delle etichette."""
    if 'etichetta' not in df.columns:
        logging.warning("Colonna 'etichetta' non trovata per calcolare la distribuzione.")
        return None
    return df['etichetta'].value_counts(normalize=True)

def get_average_sentence_length(df: pd.DataFrame, text_column: str = 'testo'):
    """Calcola la lunghezza media delle frasi (numero di parole)."""
    if text_column not in df.columns:
        logging.warning(f"Colonna '{text_column}' non trovata per calcolare la lunghezza media.")
        return None
    return df[text_column].apply(lambda x: len(str(x).split())).mean()

def get_top_n_items(df: pd.DataFrame, column_name: str, n: int = 10):
    """Restituisce i top N item più frequenti da una colonna."""
    if column_name not in df.columns:
        logging.warning(f"Colonna '{column_name}' non trovata per calcolare i top items.")
        return None
    return df[column_name].value_counts().nlargest(n)

if __name__ == '__main__':
    # Esempio di utilizzo (creare un dummy CAMI_dataset_v2.csv per testare)
    # Questo blocco non verrà eseguito se importato, solo se si esegue direttamente data_utils.py
    
    # Creazione di un dataset fittizio per il test
    dummy_data = {
        'testo': [
            "La vita è un viaggio.", "Il tempo è denaro.", "Quel professore è una volpe.", 
            "Il cielo è blu.", "Il gatto dorme sul tappeto.", "La sua voce era musica.",
            "La vita è un viaggio.", # Duplicato
            None, # Testo vuoto
            "Un cuore di ghiaccio", "Una pioggia di critiche"
        ],
        'etichetta': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1], # 1 per metafora, 0 per non-metafora
        'argomento': ['vita', 'tempo', 'professore', 'cielo', 'gatto', 'voce', 'vita', None, 'cuore', 'critiche'],
        'veicolo': ['viaggio', 'denaro', 'volpe', None, None, 'musica', 'viaggio', None, 'ghiaccio', 'pioggia']
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_csv_path = 'CAMI_dataset_v2_dummy.csv'
    dummy_df.to_csv(dummy_csv_path, index=False, encoding='utf-8')
    
    logging.info("Esecuzione di test per data_utils.py...")
    train_data, test_data = load_and_preprocess_data(dummy_csv_path)

    if train_data is not None and test_data is not None:
        logging.info(f"\n--- Training Data (head) ---\n{train_data.head()}")
        logging.info(f"\n--- Test Data (head) ---\n{test_data.head()}")

        logging.info(f"\n--- Statistiche dal Training Data ---")
        logging.info(f"Distribuzione etichette:\n{get_label_distribution(train_data)}")
        logging.info(f"Lunghezza media frasi: {get_average_sentence_length(train_data):.2f} parole")
        if 'argomento' in train_data.columns:
            logging.info(f"Top argomenti:\n{get_top_n_items(train_data, 'argomento', 5)}")
        if 'veicolo' in train_data.columns:
            logging.info(f"Top veicoli:\n{get_top_n_items(train_data, 'veicolo', 5)}")
    else:
        logging.error("Fallimento nel caricamento o processamento dei dati.")
    
    # Rimuovi il file dummy
    import os
    os.remove(dummy_csv_path)
    logging.info(f"File dummy {dummy_csv_path} rimosso.")
