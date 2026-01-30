# Problem overview

Classification task on dataset containing 79997 online news articles collected from multiple international news sources.

Dataset :

- dev: 79997
- eva: 20000

The following is a brief description of each of them:
- id: Unique identifier of the article.
- source: News outlet or publisher of the article.
- title: Title of the news article.
- article: Full textual content of the article.
- page_rank: Page rank associated with the article source.
- timestamp: Date and time of publication.
- label: Target label associated with the article (used for the classification tasks).

Label divided like so:
- International News: 0
- Business: 1
- Technology: 2
- Entertainment: 3
- Sports: 4
- General News: 5
- Health: 6


**Distribution**


(label
 0    38.855492
 1    28.620797
 2    47.573515
 3    35.996592
 4    36.675764
 5    25.811691
 6    32.955190
 Name: article_word_count, dtype: float64,
 label
 0    7.274531
 1    6.759350
 2    7.327659
 3    7.146938
 4    6.675881
 5    6.425879
 6    7.248227
 Name: title_word_count, dtype: float64)

# Proposed approach
## Data preprocessing
- Cleaning html:
  - HTML Cleaning Process Summary
    Input Validation: Ensures input is a string, returning empty if invalid.
    Metadata Extraction: Saves the <meta name="description"> content for SEO context.
    Boilerplate Removal: Deletes script, style, header, footer, and nav tags.
    Comment Stripping: Removes all hidden HTML comments.
    Image information extrapolation: We saw that meaningfull informations where contained also inside html tags, so it replaces <img> tags with their alt text to preserve meaning.
    Text Extraction: Converts remaining DOM to raw text using space separators.
    Whitespace Normalization: Merges the meta-description and collapses all whitespace into single spaces.
- removing samples with : Title, article and source duplicated
- Stemming ????? to verify if is really better.
- Vectorization of (text = article + )

## Model selection
## Hyperparameters tuning
