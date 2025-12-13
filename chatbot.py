import streamlit as st
from groq import Groq
from supabase import create_client, Client
from datetime import datetime
import hashlib

# Initialize Groq client
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Initialize Supabase client
supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)

# Page config
st.set_page_config(page_title="Simple Chatbot", page_icon="🤖", layout="wide")

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# Helper functions
def login_user(username, password):
    try:
        response = supabase.table("users").select("*").eq("username", username).eq("password", password).execute()
        if response.data:
            return response.data[0]
    except Exception as e:
        st.error(f"Login error: {e}")
    return None

def register_user(username, password):
    try:
        response = supabase.table("users").insert({
            "username": username,
            "password": password,
            "is_admin": False
        }).execute()
        return True
    except Exception as e:
        st.error(f"Registration error: {e}")
        return False

def get_user_sessions(user_id):
    try:
        response = supabase.table("chat_sessions").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
        return []

def get_all_sessions():
    try:
        response = supabase.table("chat_sessions").select("*, users(username)").order("updated_at", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Error fetching all sessions: {e}")
        return []

def create_session(user_id, session_name):
    try:
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        response = supabase.table("chat_sessions").insert({
            "session_id": session_id,
            "user_id": user_id,
            "session_name": session_name,
            "messages": []
        }).execute()
        return session_id
    except Exception as e:
        st.error(f"Error creating session: {e}")
        return None

def load_session(session_id):
    try:
        response = supabase.table("chat_sessions").select("*").eq("session_id", session_id).execute()
        if response.data:
            return response.data[0]["messages"]
    except Exception as e:
        st.error(f"Error loading session: {e}")
    return []

def save_session(session_id, messages):
    try:
        supabase.table("chat_sessions").update({
            "messages": messages,
            "updated_at": datetime.now().isoformat()
        }).eq("session_id", session_id).execute()
    except Exception as e:
        st.error(f"Error saving session: {e}")

def delete_session(session_id):
    try:
        supabase.table("chat_sessions").delete().eq("session_id", session_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting session: {e}")
        return False

def rename_session(session_id, new_name):
    try:
        supabase.table("chat_sessions").update({
            "session_name": new_name
        }).eq("session_id", session_id).execute()
        return True
    except Exception as e:
        st.error(f"Error renaming session: {e}")
        return False

def generate_session_name(first_message):
    words = first_message.split()[:4]
    return " ".join(words) if words else "New Chat"

# Login/Register page
if not st.session_state.user:
    st.title("🤖 Simple Chatbot - Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

            if submit:
                user = login_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register = st.form_submit_button("Register")

            if register:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_username) < 3:
                    st.error("Username must be at least 3 characters")
                elif len(new_password) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    if register_user(new_username, new_password):
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists")

# Admin Panel
elif st.session_state.user["is_admin"]:
    st.title("👑 Admin Panel")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["All Sessions", "Users", "Chat Interface"])

    with tab1:
        st.subheader("📊 All User Sessions")
        all_sessions = get_all_sessions()

        if all_sessions:
            st.write(f"**Total Sessions: {len(all_sessions)}**")

            for session in all_sessions:
                col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])

                with col1:
                    st.write(f"**{session['session_name']}**")
                with col2:
                    user_info = session.get('users', {})
                    username = user_info.get('username', 'Unknown') if isinstance(user_info, dict) else 'Unknown'
                    st.write(f"User: {username}")
                with col3:
                    st.write(f"Messages: {len(session['messages'])}")
                with col4:
                    if st.button("👁️", key=f"view_{session['session_id']}", help="View messages"):
                        st.session_state[f"viewing_{session['session_id']}"] = not st.session_state.get(f"viewing_{session['session_id']}", False)
                        st.rerun()
                with col5:
                    if st.button("🗑️", key=f"del_admin_{session['session_id']}"):
                        if delete_session(session['session_id']):
                            st.success("Deleted!")
                            st.rerun()

                # Show messages if viewing
                if st.session_state.get(f"viewing_{session['session_id']}", False):
                    with st.expander("💬 Conversation", expanded=True):
                        if session['messages']:
                            for idx, msg in enumerate(session['messages']):
                                role_icon = "👤" if msg['role'] == 'user' else "🤖"
                                st.markdown(f"**{role_icon} {msg['role'].title()}:**")
                                st.markdown(msg['content'])
                                if idx < len(session['messages']) - 1:
                                    st.markdown("---")
                        else:
                            st.info("No messages in this session")
                
                st.divider()
        else:
            st.info("No sessions found")

    with tab2:
        st.subheader("👥 All Users")

        try:
            response = supabase.table("users").select("*").order("created_at", desc=True).execute()
            all_users = response.data

            if all_users:
                st.write(f"**Total Users: {len(all_users)}**")

                for user in all_users:
                    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])

                    with col1:
                        admin_badge = "👑 " if user['is_admin'] else ""
                        st.write(f"**{admin_badge}{user['username']}**")
                    with col2:
                        st.write(f"Password: `{user['password']}`")
                    with col3:
                        st.write(f"Created: {user['created_at'][:10]}")
                    with col4:
                        # Count user's sessions
                        user_sessions = [s for s in get_all_sessions() if s['user_id'] == user['id']]
                        st.write(f"Sessions: {len(user_sessions)}")
                    with col5:
                        if not user['is_admin']:
                            if st.button("🗑️", key=f"del_user_{user['id']}"):
                                try:
                                    supabase.table("users").delete().eq("id", user['id']).execute()
                                    st.success("User deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")

                    st.divider()
            else:
                st.info("No users found")
        except Exception as e:
            st.error(f"Error fetching users: {e}")

    with tab3:
        st.subheader("💬 Admin Chat")
        # Admin can also use chat interface
        with st.sidebar:
            st.header("⚙️ Settings")

            model = st.selectbox(
                "Select Model",
                ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "qwen/qwen3-32b"],
                index=0
            )

            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.slider("Max Tokens", 256, 8192, 1024, 256)
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)

        # Display chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your message here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=True
                    )

                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Regular User Interface
else:
    st.title("🤖 Simple Chatbot")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Welcome, **{st.session_state.user['username']}**!")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        model = st.selectbox(
            "Select Model",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "qwen/qwen3-32b"],
            index=0
        )

        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens", 256, 8192, 1024, 256)
        top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)

        st.divider()

        # Session management
        st.subheader("💬 Chat Sessions")

        if st.button("➕ New Chat Session"):
            session_name = "New Chat"
            session_id = create_session(st.session_state.user["id"], session_name)
            if session_id:
                st.session_state.current_session_id = session_id
                st.session_state.messages = []
                st.rerun()

        # Load existing sessions
        sessions = get_user_sessions(st.session_state.user["id"])

        if sessions:
            st.write("**Your Sessions:**")
            for session in sessions:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    if st.button(f"📝 {session['session_name']}", key=f"load_{session['session_id']}"):
                        st.session_state.current_session_id = session['session_id']
                        st.session_state.messages = session['messages']
                        st.rerun()

                with col2:
                    if st.button("✏️", key=f"edit_{session['session_id']}"):
                        st.session_state[f"editing_{session['session_id']}"] = True
                        st.rerun()

                with col3:
                    if st.button("🗑️", key=f"del_{session['session_id']}"):
                        if delete_session(session['session_id']):
                            if st.session_state.current_session_id == session['session_id']:
                                st.session_state.current_session_id = None
                                st.session_state.messages = []
                            st.rerun()

                # Rename functionality
                if st.session_state.get(f"editing_{session['session_id']}", False):
                    new_name = st.text_input("New name:", value=session['session_name'], key=f"name_{session['session_id']}")
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("💾", key=f"save_{session['session_id']}"):
                            if rename_session(session['session_id'], new_name):
                                st.session_state[f"editing_{session['session_id']}"] = False
                                st.rerun()
                    with col_cancel:
                        if st.button("❌", key=f"cancel_{session['session_id']}"):
                            st.session_state[f"editing_{session['session_id']}"] = False
                            st.rerun()

        st.divider()

        if st.button("🗑️ Clear Current Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.session_state.current_session_id:
            st.info(f"**Current:** {st.session_state.current_session_id.split('_', 1)[1]}")

    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Auto-create session if none exists
        if not st.session_state.current_session_id:
            session_name = generate_session_name(prompt)
            session_id = create_session(st.session_state.user["id"], session_name)
            st.session_state.current_session_id = session_id

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=True
                )

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                # Save to database
                if st.session_state.current_session_id:
                    save_session(st.session_state.current_session_id, st.session_state.messages)

            except Exception as e:
                st.error(f"Error: {str(e)}")
