Ecco un **cheat sheet** organizzato per librerie, contenente i principali metodi estratti dai materiali forniti, con le relative **signature** e **spiegazioni testuali**.

### **NumPy (Libreria per il calcolo numerico)**

*   **`np.array(object, dtype=None)`**: Crea un array NumPy a partire da una lista o un oggetto simile; il tipo di dato può essere specificato o inferito.
*   **`np.zeros(shape)`**: Crea un array di dimensioni specificate interamente riempito di zeri.
*   **`np.ones(shape)`**: Crea un array di dimensioni specificate interamente riempito di uno.
*   **`np.full(shape, value)`**: Crea un array della forma specificata riempiendolo con un valore prestabilito.
*   **`np.linspace(start, stop, num)`**: Genera `num` campioni equispaziati tra `start` e `stop` (inclusi).
*   **`np.arange(start, stop, step)`**: Crea una sequenza di numeri da `start` a `stop` (escluso) con un determinato intervallo (`step`).
*   **`np.random.normal(mean, std, shape)`**: Genera un array di dati casuali che seguono una distribuzione normale con media e deviazione standard specificate.
*   **`np.random.random(shape)`**: Genera numeri casuali distribuiti uniformemente nell'intervallo.
*   **`np.abs(x)`, `np.exp(x)`, `np.log(x)`, `np.sin(x)`**: Funzioni universali (Ufuncs) che applicano operazioni matematiche (valore assoluto, esponenziale, logaritmo, seno) a ogni singolo elemento dell'array.
*   **`np.min(x)`, `np.max(x)`, `np.mean(x)`, `np.std(x)`, `np.sum(x)`**: Funzioni di aggregazione che calcolano rispettivamente il minimo, massimo, media, deviazione standard o somma degli elementi (anche lungo un asse specifico).
*   **`np.argmin(x)`, `np.argmax(x)`**: Restituiscono l'indice (la posizione) dell'elemento minimo o massimo nell'array.
*   **`np.sort(x, axis=-1)`**: Crea una copia ordinata dell'array lungo l'asse indicato (per riga di default).
*   **`x @ y`**: Esegue il prodotto scalare tra vettori o la moltiplicazione tra matrici.
*   **`np.concatenate((a1, a2, ...), axis=0)`**: Unisce una sequenza di array lungo un asse esistente.
*   **`np.hstack((a1, a2))`, `np.vstack((a1, a2))`**: Eseguono la concatenazione orizzontale (per riga) o verticale (per colonna, utile per aggiungere nuovi assi ai vettori 1D).
*   **`np.split(arr, indices_or_sections, axis=0)`**: Divide un array in più sotto-array in base a un numero intero o a posizioni specifiche.
*   **`array.reshape(shape)`**: Cambia la forma dell'array (es. da vettore a matrice) senza modificarne i dati; una dimensione può essere `-1` per essere inferita automaticamente.
*   **`array.copy()`**: Crea una copia fisica dei dati dell'array (utile per evitare che le modifiche su una "view" influenzino l'array originale).

---

### **Pandas (Strutture dati e analisi dati)**

*   **`pd.Series(data, index=None)`**: Crea una sequenza unidimensionale di elementi omogenei con un indice esplicito.
*   **`pd.DataFrame(data, index=None, columns=None)`**: Crea una struttura dati bidimensionale (tabella) dove le colonne possono essere viste come oggetti Series.
*   **`pd.read_csv(filepath, sep=',', skiprows=0, na_values=None)`**: Carica dati da un file CSV in un DataFrame, permettendo di specificare delimitatore e gestione dei valori nulli.
*   **`df.to_csv(path, sep=',', index=True)`**: Salva il contenuto del DataFrame in un file CSV.
*   **`df.loc[row_label, col_label]`**: Permette l'accesso a righe e colonne utilizzando etichette esplicite o maschere booleane.
*   **`df.iloc[row_index, col_index]`**: Permette l'accesso a righe e colonne tramite indici interi impliciti (posizione).
*   **`df.drop(columns=[...], inplace=False)`**: Rimuove una o più colonne specificate dal DataFrame.
*   **`df.rename(columns={'old': 'new'})`**: Rinomina le colonne del DataFrame usando un dizionario di mappatura.
*   **`df.isnull()`, `df.notnull()`**: Restituiscono maschere booleane che indicano la presenza (o assenza) di valori nulli (NaN/None).
*   **`df.dropna(axis=0, how='any')`**: Rimuove righe o colonne che contengono valori mancanti.
*   **`df.fillna(value, method=None)`**: Riempie i valori nulli con un valore specifico o una logica di propagazione (`ffill`, `bfill`).
*   **`pd.concat((objs), axis=0, ignore_index=False)`**: Concatena oggetti Pandas (Series o DataFrame) preservando gli indici o resettandoli.
*   **`pd.merge(left, right, on=None, left_index=False, right_index=False)`**: Unisce due DataFrame basandosi su colonne comuni o sui loro indici (operazione di join).
*   **`df.groupby(by)`**: Raggruppa i dati per una o più colonne per applicare successivamente funzioni di aggregazione o trasformazione.
*   **`df.pivot_table(values, index, columns, aggfunc)`**: Genera una tabella pivot per analizzare le relazioni tra diverse variabili del dataset.
*   **`df.reset_index()`**: Converte l'indice attuale in colonne del DataFrame e crea un nuovo indice numerico progressivo.
*   **`df.set_index(keys)`**: Imposta una o più colonne esistenti come nuovo indice (anche Multi-Index).
*   **`series.unstack()`, `df.stack()`**: Trasformano strutture con Multi-Index, passando da un formato a gerarchia di righe a uno a colonne e viceversa.

---

### **Matplotlib (Visualizzazione dati)**

*   **`plt.subplots(nrows=1, ncols=1, figsize=None)`**: Crea una nuova figura e un set di assi (Axes) per il disegno; `figsize` definisce le dimensioni in pollici.
*   **`ax.plot(x, y, c=None, linestyle=None, marker=None, label=None)`**: Disegna un grafico a linee con coordinate x e y; permette di personalizzare colore, stile della linea e marcatori.
*   **`ax.scatter(x, y, c=None, cmap=None, s=None)`**: Crea un grafico a dispersione (punti) dove ogni punto può avere colore (`c`) e dimensione (`s`) variabili.
*   **`ax.bar(x, height, width=0.8, tick_label=None)`**: Disegna un grafico a barre verticali; `tick_label` permette di etichettare le barre sull'asse X.
*   **`ax.hist(x, bins=None, alpha=None)`**: Genera un istogramma per visualizzare la distribuzione di frequenza di un set di dati.
*   **`ax.legend(loc=None)`**: Aggiunge una legenda basata sulle etichette (`label`) assegnate ai grafici.
*   **`ax.set_xticks(ticks)`, `ax.set_xticklabels(labels)`**: Imposta manualmente le posizioni e i testi delle etichette sull'asse delle ascisse.
*   **`plt.tight_layout()`**: Regola automaticamente i parametri del sottomodulo per far sì che i grafici rientrino correttamente nell'area della figura.
*   **`plt.subplot_mosaic(layout)`**: Crea un layout complesso di sottomoduli utilizzando una rappresentazione testuale ASCII.
*   **`fig.savefig(fname)`**: Salva la figura generata in un file immagine (es. PNG, PDF, JPG).

---

### **Scikit-learn (Machine Learning)**

#### **Preprocessing & Feature Extraction**
*   **`MinMaxScaler()`**: Scala le feature numeriche in un intervallo predefinito, solitamente.
*   **`StandardScaler()`**: Standardizza le feature trasformandole in punteggi z (media 0 e varianza 1).
*   **`OneHotEncoder(sparse_output=True)`**: Converte variabili categoriali nominali in una rappresentazione binaria "one-hot".
*   **`OrdinalEncoder(categories='auto')`**: Converte variabili categoriali ordinali in valori interi rispettando l'ordine logico.
*   **`ColumnTransformer(transformers, remainder='drop')`**: Applica trasformatori diversi a colonne specifiche di una matrice di feature.
*   **`SimpleImputer(strategy='mean')`**: Gestisce i valori mancanti sostituendoli con la media, mediana, valore più frequente o una costante.
*   **`KNNImputer(n_neighbors=5)`**: Imputa i valori mancanti basandosi sulla media dei valori dei $k$ vicini più prossimi.
*   **`CountVectorizer()`**: Trasforma documenti testuali in vettori di conteggio delle parole.
*   **`TfidfVectorizer(stop_words=None)`**: Converte testo in vettori pesati secondo la metrica TF-IDF per penalizzare le parole troppo comuni.
*   **`PCA(n_components=None)`**: Riduce la dimensionalità dei dati proiettandoli verso le direzioni di massima varianza.

#### **Modelli (Estimators)**
*   **`DecisionTreeClassifier(max_depth=None)`**: Implementa un albero di decisione per problemi di classificazione.
*   **`LinearRegression()`**: Modello per eseguire la regressione lineare ordinaria.
*   **`PolynomialFeatures(degree=2)`**: Genera nuove feature combinando quelle esistenti in forma polinomiale.
*   **`Ridge(alpha=1.0)`, `Lasso(alpha=1.0)`**: Modelli di regressione lineare con regolarizzazione (L2 per Ridge, L1 per Lasso) per prevenire l'overfitting.

#### **Workflow & Valutazione**
*   **`model.fit(X, y)`**: Addestra il modello (impara i parametri dai dati).
*   **`model.predict(X)`**: Utilizza il modello addestrato per prevedere le etichette o i valori su nuovi dati.
*   **`model.transform(X)`**: Applica la trasformazione appresa (solo per trasformatori) ai dati di input.
*   **`train_test_split(X, y, test_size)`**: Suddivide il dataset in sottoinsiemi di addestramento (training) e test.
*   **`make_pipeline(*steps)`**: Concatena una sequenza di trasformatori con un predittore finale in un unico oggetto.
*   **`accuracy_score(y_true, y_pred)`**: Calcola la precisione globale (frazione di previsioni corrette).
*   **`confusion_matrix(y_true, y_pred)`**: Mostra il dettaglio delle previsioni corrette e degli errori per ogni classe.
*   **`KFold(n_splits=5, shuffle=True)`**: Configura la suddivisione dei dati per la validazione incrociata (cross-validation).
*   **`cross_val_score(model, X, y, cv, scoring)`**: Esegue la cross-validation e restituisce il punteggio ottenuto per ogni partizione.
*   **`GridSearchCV(estimator, param_grid, scoring, cv)`**: Esegue una ricerca esaustiva su una griglia di parametri per trovare la combinazione migliore di iperparametri.