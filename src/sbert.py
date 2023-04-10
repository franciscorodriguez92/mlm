from sentence_transformers import SentenceTransformer
sentences = ["feminists are rubish","las feministas son basura", "eres una feminazi", "esto es una frase normal", "el futbol es buen deporte"]

model = SentenceTransformer('/data/frodriguez/data_mlm/models/sbert/sentence-transformers/stsb-xlm-r-multilingual-2023-02-01')
embeddings = model.encode(sentences)
print(embeddings)

""" [[-0.01029261 -0.75549895 -0.31791192 ... -1.1133536   0.8658858
   0.7663704 ]
 [-0.01799897 -0.75897866 -0.30640045 ... -1.1192538   0.85696936
   0.7714925 ]
 [-0.01946123 -0.75272083 -0.3023723  ... -1.1241705   0.85667104
   0.7738365 ]
 [ 0.44341308  0.06955273  1.8015159  ...  0.48044804 -1.0221393
   0.10151388]
 [ 0.45787287  0.05965161  1.8125309  ...  0.5030064  -1.0073416
   0.10007063]] """