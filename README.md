# PokemonTCG Card Detector

A Python project for detecting and identifying Pokémon TCG cards using computer vision and the Pokémon TCG API.

## Features

- Detects Pokémon cards from webcam
- Matches detected cards to a local database using perceptual hashing
- Fetches and displays card market prices using the Pokémon TCG API

## Setup

1. Clone the repository.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Create a `.env` file in the project root with your API key:
    ```
    POKEMON_API_KEY=your_api_key_here
    ```

## Usage

You have two options to use this project:

### Option 1: Use the Prebuilt Database

If you want to get started quickly, use the already included `card_database.pkl` file and run the card matcher:

```bash
python card_matcher.py
```
### Option 2: Generate Your Own Database

If you want to generate your own database, you can run the following scripts:

- To fetch and download card images:
    ```bash
    python fetch_cards.py
    ```
- To generate the card database:
    ```bash
    python card_hasher.py
    ''''


