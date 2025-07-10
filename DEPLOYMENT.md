# Deployment Guide

This guide covers deploying the Pokemon Card Detection and Collection System to various platforms.

## Local Development

### Prerequisites
- Python 3.8+
- Webcam
- Supabase account
- Pokemon TCG API key (optional)

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see `.env.example`)
4. Run the application: `python webapp/app.py`

## Cloud Deployment

### Heroku

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Set environment variables**
   ```bash
   heroku config:set SUPABASE_URL=your_supabase_url
   heroku config:set SUPABASE_KEY=your_supabase_key
   heroku config:set POKEMON_API_KEY=your_pokemon_api_key
   heroku config:set FLASK_SECRET_KEY=your_secret_key
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

### Railway

1. **Connect your GitHub repository**
2. **Set environment variables** in Railway dashboard
3. **Deploy automatically** on push to main branch

### DigitalOcean App Platform

1. **Create new app** from GitHub repository
2. **Set environment variables** in app settings
3. **Deploy** automatically

## Environment Variables

Required environment variables:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
POKEMON_API_KEY=your_pokemon_tcg_api_key
FLASK_SECRET_KEY=your_flask_secret_key
```

## Database Setup

1. **Create Supabase project**
2. **Run SQL schema** in Supabase SQL Editor:
   ```sql
   -- Run the contents of webapp/supabase_schema.sql
   ```
3. **Update environment variables** with Supabase credentials

## Production Considerations

### Security
- Use strong Flask secret key
- Enable HTTPS in production
- Set up proper CORS policies
- Use environment variables for sensitive data

### Performance
- Enable gzip compression
- Use CDN for static assets
- Optimize database queries
- Consider caching strategies

### Monitoring
- Set up logging
- Monitor application performance
- Track API usage
- Set up error reporting

## Troubleshooting

### Common Issues

1. **Webcam not working**
   - Check browser permissions
   - Ensure HTTPS in production
   - Test with different browsers

2. **Database connection errors**
   - Verify Supabase credentials
   - Check RLS policies
   - Ensure proper table structure

3. **API rate limiting**
   - Implement caching
   - Add rate limiting
   - Monitor API usage

### Support

For deployment issues:
1. Check application logs
2. Verify environment variables
3. Test locally first
4. Check platform-specific documentation 