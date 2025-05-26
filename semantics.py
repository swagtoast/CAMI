import nltk
from nltk.corpus import wordnet as wn
import logging

# Impostazioni di logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessari per NLTK WordNet (da eseguire una volta)
# In un notebook o script principale, puoi usare:
# nltk.download('wordnet')
# nltk.download('omw-1.4') # Open Multilingual WordNet per l'italiano

def _get_synsets_ita(parola: str):
    """Funzione helper per ottenere i synset italiani di una parola."""
    # Tentativo con forma originale
    synsets = wn.synsets(parola, lang='ita')
    if not synsets:
        # Tentativo con forma minuscola (se non già)
        synsets = wn.synsets(parola.lower(), lang='ita')
    # Qui si potrebbe aggiungere la lemmatizzazione se si ha un lemmatizzatore italiano affidabile
    # Esempio (richiede un lemmatizzatore, es. da spaCy o altri):
    # from some_lemmatizer import lemmatize_italian
    # lemma = lemmatize_italian(parola.lower())
    # if lemma != parola.lower():
    #     synsets.extend(wn.synsets(lemma, lang='ita'))
    # Rimuovi duplicati mantenendo l'ordine se necessario, anche se wn.synsets non dovrebbe darne molti
    # Per ora, ci affidiamo alla ricerca diretta e al lowercase.
    return synsets

def distanza_semantica(argomento: str, veicolo: str, similarity_metric: str = 'path') -> float | None:
    """
    Calcola una misura di similarità semantica tra due parole italiane (argomento e veicolo)
    utilizzando NLTK WordNet (OMW).

    Args:
        argomento (str): La prima parola (es. "vita").
        veicolo (str): La seconda parola (es. "viaggio").
        similarity_metric (str): Metrica di similarità da usare. 
                                 'path' per path_similarity, 'wup' per wup_similarity.

    Returns:
        float | None: La massima similarità trovata tra le coppie di synset,
                      o 0.0 se non ci sono synset comuni o le parole non sono trovate.
                      Restituisce None se una delle parole non ha synset.
    """
    if not argomento or not veicolo:
        logging.warning("Argomento o veicolo mancanti per calcolo distanza semantica.")
        return None

    synsets_arg = _get_synsets_ita(argomento)
    synsets_vec = _get_synsets_ita(veicolo)

    if not synsets_arg:
        logging.debug(f"Nessun synset trovato per l'argomento '{argomento}' in WordNet (italiano).")
        # Fallback: si potrebbe tentare con WordNet inglese se la parola è simile o tradotta,
        # oppure usare BabelNet se disponibile e configurato.
        # Per ora, restituiamo None per indicare fallimento.
        return None
    if not synsets_vec:
        logging.debug(f"Nessun synset trovato per il veicolo '{veicolo}' in WordNet (italiano).")
        return None

    max_similarity = 0.0
    found_similarity = False

    for s_arg in synsets_arg:
        for s_vec in synsets_vec:
            # Assicurati che entrambi i synset siano dello stesso PoS (Part of Speech) o che la metrica lo gestisca.
            # path_similarity e wup_similarity di solito funzionano meglio tra nomi e nomi, verbi e verbi.
            # Se i PoS sono diversi, la similarità potrebbe essere bassa o None.
            # WordNetpy restituisce None se la similarità non può essere calcolata (es. POS diversi per path_similarity)
            
            current_similarity = None
            if similarity_metric == 'path':
                current_similarity = s_arg.path_similarity(s_vec)
            elif similarity_metric == 'wup':
                current_similarity = s_arg.wup_similarity(s_vec)
            else:
                logging.warning(f"Metrica di similarità '{similarity_metric}' non supportata. Uso 'path'.")
                current_similarity = s_arg.path_similarity(s_vec)

            if current_similarity is not None and current_similarity > max_similarity:
                max_similarity = current_similarity
                found_similarity = True
    
    return max_similarity if found_similarity else 0.0 # Restituisce 0.0 se nessuna coppia ha dato similarità valida

def stima_concretezza(parola: str) -> bool | None:
    """
    Stima se una parola è concreta o astratta basandosi sulla sua iperonimia in WordNet.
    Una parola è considerata concreta se uno dei suoi synset ha "physical_entity.n.01"
    come iperonimo.

    Args:
        parola (str): La parola da analizzare.

    Returns:
        bool | None: True se la parola è stimata concreta, False se astratta (o non fisica),
                     None se la parola non è trovata o non si può determinare.
    """
    if not parola:
        logging.warning("Parola mancante per stima concretezza.")
        return None

    synsets_parola = _get_synsets_ita(parola)
    if not synsets_parola:
        logging.debug(f"Nessun synset trovato per '{parola}' in WordNet (italiano) per stima concretezza.")
        # Fallback: Potrebbe essere necessario un dizionario di concretezza esterno o BabelNet.
        return None

    try:
        physical_entity_synset = wn.synset('physical_entity.n.01') # Questo è un synset "ancora" universale
    except Exception as e:
        logging.error(f"Impossibile trovare il synset 'physical_entity.n.01': {e}")
        return None # Non possiamo procedere senza questo riferimento

    for s_parola in synsets_parola:
        # Controlla se physical_entity_synset è un iperonimo (diretto o indiretto)
        # Il metodo closure genera tutti gli iperonimi ricorsivamente.
        hypernym_closure = set(s_parola.closure(lambda s: s.hypernyms()))
        if physical_entity_synset in hypernym_closure:
            return True  # Trovata entità fisica, quindi concreta

    return False # Nessun synset della parola è risultato essere una 'physical_entity'

if __name__ == '__main__':
    # Esempio di utilizzo (assicurati di aver scaricato i dati NLTK)
    # In Colab, eseguire prima in una cella:
    # import nltk
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    
    logging.info("Esecuzione di test per semantics.py...")

    # Esempi Distanza Semantica
    print("\n--- Distanza Semantica (Path Similarity) ---")
    parole_coppie = [
        ("cane", "gatto"),      # Simili, stessa categoria
        ("automobile", "nave"), # Simili, mezzi di trasporto
        ("uomo", "pensiero"),   # Distanti, concreto vs. astratto
        ("re", "corona"),       # Relazionati ma non strettamente simili tassonomicamente
        ("parola_inesistente_xyz", "test") # Test parola non trovata
    ]
    for arg, vec in parole_coppie:
        dist_path = distanza_semantica(arg, vec, similarity_metric='path')
        dist_wup = distanza_semantica(arg, vec, similarity_metric='wup')
        print(f"Path Sim ('{arg}', '{vec}'): {dist_path}")
        print(f"WUP Sim  ('{arg}', '{vec}'): {dist_wup}")

    # Esempi Stima Concretezza
    print("\n--- Stima Concretezza ---")
    parole_singole = [
        "tavolo", "sedia",      # Concreti
        "idea", "amore", "pensiero", # Astratti
        "gatto", "albero",      # Concreti (esseri viventi/oggetti fisici)
        "entità",              # Potrebbe essere ambiguo, dipende dal synset
        "valore",              # Astratto
        "parola_inesistente_xyz" # Test parola non trovata
    ]
    for p in parole_singole:
        concretezza = stima_concretezza(p)
        print(f"Concretezza di '{p}': {concretezza}")

    # Test con argomenti e veicoli di metafore note
    print("\n--- Analisi Semantica Metafore Note ---")
    metafore_note = [
        {"argomento": "vita", "veicolo": "viaggio"},
        {"argomento": "discussione", "veicolo": "guerra"},
        {"argomento": "tempo", "veicolo": "denaro"},
        {"argomento": "occhi", "veicolo": "stelle"}
    ]
    for metafora in metafore_note:
        arg = metafora["argomento"]
        vec = metafora["veicolo"]
        dist = distanza_semantica(arg, vec)
        conc_arg = stima_concretezza(arg)
        conc_vec = stima_concretezza(vec)
        print(f"Metafora: '{arg}' è '{vec}'")
        print(f"  Distanza Semantica: {dist}")
        print(f"  Concretezza Argomento ('{arg}'): {conc_arg}")
        print(f"  Concretezza Veicolo ('{vec}'): {conc_vec}")
        print("-" * 20)

    # Commenti aggiuntivi:
    # - Se WordNet non trova il lemma italiano, potrebbe essere necessario un fallback
    #   a un dizionario inglese (se la parola è un prestito o simile) o a risorse più ampie
    #   come BabelNet, che però richiedono setup più complessi (API keys, ecc.).
    # - La lemmatizzazione accurata in italiano prima di interrogare WordNet può migliorare
    #   significativamente i risultati, specialmente per forme verbali flesse o plurali.