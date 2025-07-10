# Pokemon Card Detection and Collection System

A full-stack web application that uses computer vision to detect Pokemon cards via webcam, match them against a database, and manage your card collection with real-time price tracking.

## Features

- **Real-time Card Detection**: Uses OpenCV and contour detection to identify Pokemon cards through webcam
- **Card Matching**: Perceptual hashing (pHash) algorithm to match detected cards against a database
- **Web-based Collection Management**: Modern web interface built with Flask and Tailwind CSS
- **User Authentication**: Secure user accounts with Supabase authentication
- **Price Tracking**: Integration with Pokemon TCG API for real-time card prices
- **Deck Management**: Add, remove, and organize your Pokemon card collection
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML, CSS (Tailwind), JavaScript
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth
- **Computer Vision**: OpenCV, PIL
- **Image Processing**: Perceptual Hashing (pHash)
- **API Integration**: Pokemon TCG API

## Project Structure

```
pokemon-card-detector/
├── webapp/                    # Main Flask application
│   ├── app.py                # Flask server and card detection logic
│   ├── templates/            # HTML templates
│   ├── requirements.txt      # Python dependencies
│   └── README.md            # Webapp setup instructions
├── card_matcher.py           # Standalone card detection script
├── card_hasher.py           # Database creation utility
├── card_database.pkl        # Card database (not in repo)
├── requirements.txt         # Main project dependencies
└── README.md               # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- Webcam
- Pokemon TCG API key (optional, for price tracking)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pokemon-card-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the web application**
   ```bash
   cd webapp
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp env_example.txt .env
   # Edit .env with your Supabase credentials and API keys
   ```

5. **Set up the database**
   - Create a Supabase project
   - Run the SQL schema in Supabase SQL Editor
   - Update your .env file with Supabase credentials

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the application**
   - Open http://localhost:5000 in your browser
   - Register/login to start using the card detection system

## Usage

### Web Application

1. **Register/Login**: Create an account or sign in
2. **Start Camera**: Click "Start Camera" on the dashboard
3. **Detect Cards**: Point your webcam at Pokemon cards
4. **Add to Collection**: Confirm detected cards to add them to your deck
5. **Manage Collection**: View, organize, and track your cards

### Standalone Card Detection

Run the standalone script for quick card detection:
```bash
python card_matcher.py
```

## Database Setup

The application uses Supabase for:
- User authentication
- Card collection storage
- Real-time data synchronization

### Required Environment Variables

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
POKEMON_API_KEY=your_pokemon_tcg_api_key
FLASK_SECRET_KEY=your_flask_secret_key
```

## API Integration

### Pokemon TCG API
- Fetches real-time card prices
- Retrieves card images
- Provides card metadata

### Supabase
- User authentication
- Database storage
- Real-time updates

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Pokemon TCG API for card data
- Supabase for backend services
- OpenCV for computer vision capabilities
- Tailwind CSS for styling


