# Pokemon Card Manager Web Application

A beautiful web application for managing your Pokemon card collection using AI-powered card detection, built with Flask and Supabase.

## Features

- 🔐 **User Authentication** - Secure login/registration with Supabase Auth
- 📸 **AI Card Detection** - Upload photos and automatically identify Pokemon cards
- 💾 **Digital Deck Management** - Organize and track your card collection
- 💰 **Price Tracking** - Get real-time market prices for your cards
- 📊 **Collection Statistics** - View your collection value and statistics
- 🎨 **Modern UI** - Beautiful, responsive design with Tailwind CSS
- 🔒 **Secure Storage** - Your data is safely stored in Supabase

## Tech Stack

- **Backend**: Flask (Python)
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **AI**: OpenCV, ImageHash, YOLOv8
- **Deployment**: Gunicorn

## Prerequisites

- Python 3.8+
- Supabase account
- Pokemon TCG API key (optional, for price data)

## Setup Instructions

### 1. Clone and Navigate to Webapp Directory

```bash
cd webapp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Supabase

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Get your project URL and anon key from Settings > API
3. Create the following table in your Supabase database:

```sql
-- Create user_decks table
CREATE TABLE user_decks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    card_name TEXT NOT NULL,
    card_id TEXT,
    price_info TEXT,
    distance FLOAT,
    confidence TEXT,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    image_data TEXT
);

-- Enable Row Level Security
ALTER TABLE user_decks ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to see only their own cards
CREATE POLICY "Users can view own cards" ON user_decks
    FOR SELECT USING (auth.uid() = user_id);

-- Create policy to allow users to insert their own cards
CREATE POLICY "Users can insert own cards" ON user_decks
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Create policy to allow users to delete their own cards
CREATE POLICY "Users can delete own cards" ON user_decks
    FOR DELETE USING (auth.uid() = user_id);
```

### 4. Configure Environment Variables

Create a `.env` file in the webapp directory:

```bash
# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here-change-this-in-production

# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-supabase-anon-key

# Pokemon TCG API (Optional - for price data)
POKEMON_API_KEY=your-pokemon-tcg-api-key
```

### 5. Copy Card Database

Make sure your `card_database.pkl` file is in the parent directory (one level up from webapp):

```bash
# From the webapp directory
cp ../card_database.pkl ./
```

### 6. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### For Users

1. **Register/Login**: Create an account or sign in
2. **Upload Cards**: Use the upload area to add photos of your Pokemon cards
3. **View Collection**: See all your cards in the dashboard
4. **Track Value**: Monitor your collection's estimated value
5. **Manage Deck**: Add or remove cards from your collection

### For Developers

#### Project Structure

```
webapp/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/            # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Landing page
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   └── dashboard.html    # Main dashboard
├── static/               # Static files (CSS, JS, images)
└── .env                  # Environment variables
```

#### Key Features

- **Card Detection**: Uses your existing `card_matcher.py` logic
- **Image Processing**: Supports drag-and-drop and file selection
- **Real-time Updates**: Live statistics and deck management
- **Responsive Design**: Works on desktop and mobile devices

#### API Endpoints

- `GET /` - Landing page
- `GET /login` - Login page
- `POST /login` - Login API
- `GET /register` - Registration page
- `POST /register` - Registration API
- `GET /dashboard` - User dashboard
- `POST /upload-card` - Upload and process card image
- `GET /api/deck` - Get user's deck
- `POST /api/remove-card` - Remove card from deck
- `GET /api/stats` - Get collection statistics

## Deployment

### Local Development

```bash
python app.py
```

### Production Deployment

1. Install Gunicorn:
```bash
pip install gunicorn
```

2. Run with Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. For production, consider using:
   - Nginx as reverse proxy
   - Environment variables for configuration
   - HTTPS certificates
   - Database backups

## Customization

### Styling

The application uses Tailwind CSS for styling. You can customize the design by modifying the CSS classes in the HTML templates.

### Card Detection

The card detection logic is in the `process_uploaded_image()` function in `app.py`. You can modify this to use different detection algorithms or thresholds.

### Database Schema

You can extend the `user_decks` table to include additional fields like:
- Card condition
- Purchase price
- Notes
- Tags/categories

## Troubleshooting

### Common Issues

1. **Card Database Not Found**: Ensure `card_database.pkl` is in the correct location
2. **Supabase Connection Error**: Check your environment variables
3. **Image Upload Fails**: Verify file format and size
4. **Authentication Issues**: Check Supabase Auth configuration

### Debug Mode

For development, the app runs in debug mode by default. For production, set:

```python
app.run(debug=False)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For support or questions, please open an issue in the repository. 