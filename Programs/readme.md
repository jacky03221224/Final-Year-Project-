# Programs

## cleaning.py
This program is based on the [cleaning-prompt](../Prompts) and is responsible for preprocessing and standardising raw news datasets before further analysis.  
It performs the following tasks:
1. Remove news entries with missing values
2. Remove news entries with promotional content from the Finhub API
3. Remove identical news entries
4. Formatting

## deduplicate_TFIDF.py
This program is based on the [deduplicate-prompt](../Prompts) and is designed to remove semantically similar news entries from the cleaned datasets using a TF-IDF-based similarity approach.  
It performs the following tasks:
1. Group news into Monday-Sunday weekly buckets
2. Use TF-IDF to measure the similarity between news entries
3. Group news entries into clusters with cosine similarity greater than 0.7
4. Retain only one news entry with the longest headline + summary from each cluster

## relevance.py
This program is based on the [relevance-prompt](../Prompts).  
It requires a config file and a CSV file as input.  
It uses keyword matching to assign labels and scores to each financial news data.  
We did not choose to use any Machine learning model because we do not have time for labeling the dataset manually.  
It ranges from [0,1], where:
* Directly Related: 0.70 - 1.00
* Indirectly Related: 0.30 - 0.69
* Unrelated: 0.00 - 0.29

