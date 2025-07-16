from flask import Blueprint, request, jsonify, session, render_template, redirect, url_for
from supabase import create_client, Client
import os
from datetime import datetime

bp = Blueprint('friends', __name__)

supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')
supabase: Client = create_client(supabase_url, supabase_key)

@bp.route('/friends', methods=['GET'])
def friends_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('friends.html')

@bp.route('/messages', methods=['GET'])
def messages_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Get friend parameter if provided
    friend_id = request.args.get('friend')
    
    # If friend_id is provided, verify they are friends
    if friend_id:
        user_id = session['user']['id']
        try:
            friendship = supabase.table('friendships').select('*').or_(f'user_id.eq.{user_id},friend_id.eq.{user_id}').execute()
            # Check if the specific friendship exists
            is_friends = any(
                (row['user_id'] == user_id and row['friend_id'] == friend_id) or 
                (row['user_id'] == friend_id and row['friend_id'] == user_id) 
                for row in friendship.data or []
            )
            if not is_friends:
                friend_id = None  # Reset if not friends
        except:
            friend_id = None  # Reset on error
    
    return render_template('messages.html', selected_friend=friend_id)

@bp.route('/api/friends/add', methods=['POST'])
def add_friend():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    data = request.get_json()
    friend_email = data.get('friend_email')
    user_id = session['user']['id']
    
    # Use Supabase auth admin API to find user by email
    try:
        # This requires admin privileges - we'll need to use a different approach
        # For now, let's create a simple user lookup table or use a different method
        # For development, we can create a users table in the public schema
        friend = supabase.table('users').select('id, email').eq('email', friend_email).single().execute()
        if not friend.data:
            return jsonify({'success': False, 'error': 'This email is not registered.'}), 404
        friend_id = friend.data['id']
    except Exception as e:
        # If users table doesn't exist, we'll need to create it or use a different approach
        return jsonify({'success': False, 'error': 'User lookup not available. Please create a users table in Supabase.'}), 500
    
    if friend_id == user_id:
        return jsonify({'success': False, 'error': 'Cannot add yourself as a friend.'}), 400
    
    # Check if already friends - use proper filter syntax
    friendship = supabase.table('friendships').select('*').or_(f'user_id.eq.{user_id},friend_id.eq.{user_id}').execute()
    # Filter the results to check if the specific friend relationship exists
    is_friends = any(
        (row['user_id'] == user_id and row['friend_id'] == friend_id) or 
        (row['user_id'] == friend_id and row['friend_id'] == user_id) 
        for row in friendship.data or []
    )
    if is_friends:
        return jsonify({'success': False, 'error': 'You are already friends.'}), 400
    
    # Check if a pending request exists - use proper filter syntax
    request_exists = supabase.table('friend_requests').select('*').or_(f'sender_id.eq.{user_id},receiver_id.eq.{user_id}').execute()
    # Filter the results to check if the specific request exists
    has_pending_request = any(
        (row['sender_id'] == user_id and row['receiver_id'] == friend_id) or 
        (row['sender_id'] == friend_id and row['receiver_id'] == user_id) 
        for row in request_exists.data or []
    )
    if has_pending_request:
        return jsonify({'success': False, 'error': 'A friend request is already pending.'}), 400
    
    # Check if there's a declined request that we can update
    declined_request = None
    for row in request_exists.data or []:
        if ((row['sender_id'] == user_id and row['receiver_id'] == friend_id) or 
            (row['sender_id'] == friend_id and row['receiver_id'] == user_id)) and row['status'] == 'declined':
            declined_request = row
            break
    
    # Create friend request or update declined one
    try:
        if declined_request:
            # Update the declined request to pending
            result = supabase.table('friend_requests').update({
                'status': 'pending',
                'created_at': datetime.now().isoformat()
            }).eq('id', declined_request['id']).execute()
        else:
            # Create new friend request
            result = supabase.table('friend_requests').insert({
                'sender_id': user_id,
                'receiver_id': friend_id,
                'status': 'pending'
            }).execute()
        
        return jsonify({'success': True, 'message': 'Friend request sent!'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': f'Failed to create friend request: {str(e)}'}), 400

@bp.route('/api/friends/list', methods=['GET'])
def list_friends():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    user_id = session['user']['id']
    # Get friendships where user is user_id or friend_id
    try:
        result = supabase.table('friendships').select('*').or_(f'user_id.eq.{user_id},friend_id.eq.{user_id}').execute()
        friends = []
        for row in result.data:
            fid = row['friend_id'] if row['user_id'] == user_id else row['user_id']
            try:
                user_info = supabase.table('users').select('id, email').eq('id', fid).single().execute()
                if user_info.data:
                    friends.append({'id': fid, 'email': user_info.data['email']})
            except:
                # If users table doesn't exist, just use the ID
                friends.append({'id': fid, 'email': f'User {fid[:8]}...'})
        return jsonify({'success': True, 'friends': friends}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/friends/requests', methods=['GET'])
def list_friend_requests():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    user_id = session['user']['id']
    # Incoming requests
    try:
        incoming = supabase.table('friend_requests').select('id, sender_id, created_at').eq('receiver_id', user_id).eq('status', 'pending').execute()
        incoming_requests = []
        for req in incoming.data or []:
            try:
                sender_info = supabase.table('users').select('id, email').eq('id', req['sender_id']).single().execute()
                sender_email = sender_info.data['email'] if sender_info.data else f'User {req["sender_id"][:8]}...'
            except:
                sender_email = f'User {req["sender_id"][:8]}...'
            incoming_requests.append({
                'id': req['id'],
                'sender_id': req['sender_id'],
                'sender_email': sender_email,
                'created_at': req['created_at']
            })
        # Outgoing requests
        outgoing = supabase.table('friend_requests').select('id, receiver_id, created_at').eq('sender_id', user_id).eq('status', 'pending').execute()
        outgoing_requests = []
        for req in outgoing.data or []:
            try:
                receiver_info = supabase.table('users').select('id, email').eq('id', req['receiver_id']).single().execute()
                receiver_email = receiver_info.data['email'] if receiver_info.data else f'User {req["receiver_id"][:8]}...'
            except:
                receiver_email = f'User {req["receiver_id"][:8]}...'
            outgoing_requests.append({
                'id': req['id'],
                'receiver_id': req['receiver_id'],
                'receiver_email': receiver_email,
                'created_at': req['created_at']
            })
        return jsonify({'success': True, 'incoming': incoming_requests, 'outgoing': outgoing_requests}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/friends/requests/<request_id>/accept', methods=['POST'])
def accept_friend_request(request_id):
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    user_id = session['user']['id']
    # Get the request
    try:
        req = supabase.table('friend_requests').select('*').eq('id', request_id).single().execute()
        if not req.data or req.data['receiver_id'] != user_id or req.data['status'] != 'pending':
            return jsonify({'success': False, 'error': 'Invalid or unauthorized request.'}), 400
        # Update request status
        supabase.table('friend_requests').update({'status': 'accepted'}).eq('id', request_id).execute()
        # Create friendship
        supabase.table('friendships').insert({
            'user_id': req.data['sender_id'],
            'friend_id': req.data['receiver_id']
        }).execute()
        return jsonify({'success': True, 'message': 'Friend request accepted.'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/friends/requests/<request_id>/decline', methods=['POST'])
def decline_friend_request(request_id):
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    user_id = session['user']['id']
    try:
        req = supabase.table('friend_requests').select('*').eq('id', request_id).single().execute()
        if not req.data or req.data['receiver_id'] != user_id or req.data['status'] != 'pending':
            return jsonify({'success': False, 'error': 'Invalid or unauthorized request.'}), 400
        supabase.table('friend_requests').update({'status': 'declined'}).eq('id', request_id).execute()
        return jsonify({'success': True, 'message': 'Friend request declined.'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/friends/<friend_id>/dashboard', methods=['GET'])
def friend_dashboard_api(friend_id):
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    user_id = session['user']['id']
    try:
        friendship = supabase.table('friendships').select('*').or_(f'user_id.eq.{user_id},friend_id.eq.{user_id}').execute()
        # Check if the specific friendship exists
        is_friends = any(
            (row['user_id'] == user_id and row['friend_id'] == friend_id) or 
            (row['user_id'] == friend_id and row['friend_id'] == user_id) 
            for row in friendship.data or []
        )
        if not is_friends:
            return jsonify({'success': False, 'error': 'Not friends'}), 403
        cards = supabase.table('user_decks').select('*').eq('user_id', friend_id).execute()
        return jsonify({'success': True, 'cards': cards.data}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/friends/<friend_id>', methods=['GET'])
def friend_dashboard_page(friend_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    user_id = session['user']['id']
    try:
        friendship = supabase.table('friendships').select('*').or_(f'user_id.eq.{user_id},friend_id.eq.{user_id}').execute()
        # Check if the specific friendship exists
        is_friends = any(
            (row['user_id'] == user_id and row['friend_id'] == friend_id) or 
            (row['user_id'] == friend_id and row['friend_id'] == user_id) 
            for row in friendship.data or []
        )
        if not is_friends:
            return render_template('friends.html', error='Not friends')
        cards = supabase.table('user_decks').select('*').eq('user_id', friend_id).execute()
        try:
            friend_info = supabase.table('users').select('email').eq('id', friend_id).single().execute()
            friend_email = friend_info.data['email'] if friend_info.data else f'User {friend_id[:8]}...'
        except:
            friend_email = f'User {friend_id[:8]}...'
        return render_template('friend_dashboard.html', cards=cards.data, friend_email=friend_email)
    except Exception as e:
        return render_template('friends.html', error=f'Error: {str(e)}')

# Messaging endpoints
@bp.route('/api/messages/send', methods=['POST'])
def send_message():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    receiver_id = data.get('receiver_id')
    message_text = data.get('message_text')
    user_id = session['user']['id']
    
    if not message_text or not receiver_id:
        return jsonify({'success': False, 'error': 'Missing message text or receiver'}), 400
    
    # Check if they are friends
    try:
        friendship = supabase.table('friendships').select('*').or_(f'user_id.eq.{user_id},friend_id.eq.{user_id}').execute()
        is_friends = any(
            (row['user_id'] == user_id and row['friend_id'] == receiver_id) or 
            (row['user_id'] == receiver_id and row['friend_id'] == user_id) 
            for row in friendship.data or []
        )
        if not is_friends:
            return jsonify({'success': False, 'error': 'Can only message friends'}), 403
        
        # Send message
        result = supabase.table('messages').insert({
            'sender_id': user_id,
            'receiver_id': receiver_id,
            'message_text': message_text
        }).execute()
        
        return jsonify({'success': True, 'message': 'Message sent!'}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/messages/conversations', methods=['GET'])
def get_conversations():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_id = session['user']['id']
    
    try:
        # Get all messages where user is sender or receiver
        messages = supabase.table('messages').select('*').or_(f'sender_id.eq.{user_id},receiver_id.eq.{user_id}').order('created_at', desc=True).execute()
        
        print(f"Found {len(messages.data or [])} messages for user {user_id}")  # Debug print
        
        # Group by conversation partner
        conversations = {}
        for msg in messages.data or []:
            other_user_id = msg['receiver_id'] if msg['sender_id'] == user_id else msg['sender_id']
            
            if other_user_id not in conversations:
                conversations[other_user_id] = {
                    'user_id': other_user_id,
                    'last_message': msg['message_text'],
                    'last_message_time': msg['created_at'],
                    'unread_count': 0
                }
            
            # Count unread messages
            if msg['receiver_id'] == user_id and not msg['is_read']:
                conversations[other_user_id]['unread_count'] += 1
        
        # Get user emails for conversation partners
        for conv in conversations.values():
            try:
                user_info = supabase.table('users').select('email').eq('id', conv['user_id']).single().execute()
                conv['user_email'] = user_info.data['email'] if user_info.data else f'User {conv["user_id"][:8]}...'
            except:
                conv['user_email'] = f'User {conv["user_id"][:8]}...'
        
        print(f"Returning {len(conversations)} conversations")  # Debug print
        return jsonify({'success': True, 'conversations': list(conversations.values())}), 200
    except Exception as e:
        print(f"Error in get_conversations: {e}")  # Debug print
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/messages/<other_user_id>', methods=['GET'])
def get_messages(other_user_id):
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_id = session['user']['id']
    
    try:
        # Get messages between the two users - simplified query
        messages = supabase.table('messages').select('*').or_(f'sender_id.eq.{user_id},receiver_id.eq.{user_id}').execute()
        
        # Filter messages to only include those between the two users
        filtered_messages = []
        for msg in messages.data or []:
            if ((msg['sender_id'] == user_id and msg['receiver_id'] == other_user_id) or 
                (msg['sender_id'] == other_user_id and msg['receiver_id'] == user_id)):
                filtered_messages.append(msg)
        
        # Sort by created_at
        filtered_messages.sort(key=lambda x: x['created_at'])
        
        # Mark messages as read
        supabase.table('messages').update({'is_read': True}).eq('sender_id', other_user_id).eq('receiver_id', user_id).eq('is_read', False).execute()
        
        return jsonify({'success': True, 'messages': filtered_messages}), 200
    except Exception as e:
        print(f"Error in get_messages: {e}")  # Debug print
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/messages/<other_user_id>/mark-read', methods=['POST'])
def mark_messages_read(other_user_id):
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_id = session['user']['id']
    
    try:
        # Mark messages as read
        supabase.table('messages').update({'is_read': True}).eq('sender_id', other_user_id).eq('receiver_id', user_id).eq('is_read', False).execute()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/friends/<friend_id>/email', methods=['GET'])
def get_friend_email(friend_id):
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_id = session['user']['id']
    
    try:
        # Check if they are friends
        friendship = supabase.table('friendships').select('*').or_(f'user_id.eq.{user_id},friend_id.eq.{user_id}').execute()
        is_friends = any(
            (row['user_id'] == user_id and row['friend_id'] == friend_id) or 
            (row['user_id'] == friend_id and row['friend_id'] == user_id) 
            for row in friendship.data or []
        )
        if not is_friends:
            return jsonify({'success': False, 'error': 'Not friends'}), 403
        
        # Get friend's email
        try:
            friend_info = supabase.table('users').select('email').eq('id', friend_id).single().execute()
            friend_email = friend_info.data['email'] if friend_info.data else f'User {friend_id[:8]}...'
        except:
            friend_email = f'User {friend_id[:8]}...'
        
        return jsonify({'success': True, 'email': friend_email}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@bp.route('/api/debug/messages', methods=['GET'])
def debug_messages():
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    user_id = session['user']['id']
    
    try:
        # Check if messages table exists and has data
        messages = supabase.table('messages').select('*').limit(10).execute()
        print(f"Debug: Found {len(messages.data or [])} total messages")
        
        # Check user's messages
        user_messages = supabase.table('messages').select('*').or_(f'sender_id.eq.{user_id},receiver_id.eq.{user_id}').execute()
        print(f"Debug: Found {len(user_messages.data or [])} messages for user {user_id}")
        
        return jsonify({
            'success': True, 
            'total_messages': len(messages.data or []),
            'user_messages': len(user_messages.data or []),
            'user_id': user_id,
            'sample_messages': user_messages.data[:5] if user_messages.data else []
        }), 200
    except Exception as e:
        print(f"Debug error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400 