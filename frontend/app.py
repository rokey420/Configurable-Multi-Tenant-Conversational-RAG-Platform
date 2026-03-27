# frontend/app.py
import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = 60

st.set_page_config(page_title="Multi-Tenant RAG Platform", layout="wide")
st.title("Multi-Tenant Document Chatbot")

# -----------------------------
# Session defaults
# -----------------------------
st.session_state.setdefault("mode", None)           # None / "admin" / "user"
st.session_state.setdefault("admin_token", None)    # X-Admin-Token from backend
st.session_state.setdefault("admin_username", None)

st.session_state.setdefault("user_id", None)
st.session_state.setdefault("username", None)

st.session_state.setdefault("messages", [])
st.session_state.setdefault("topics", [])
st.session_state.setdefault("topic_id", None)

# ✅ NEW: prompt state (draft vs improved + chooser)
st.session_state.setdefault("topic_name_draft", "")
st.session_state.setdefault("topic_prompt_draft", "")
st.session_state.setdefault("improved_prompt", "")
st.session_state.setdefault("prompt_choice", "Draft")  # Draft / Improved


# -----------------------------
# HTTP helpers
# -----------------------------
def api_post(path: str, payload: dict, headers: dict | None = None):
    return requests.post(f"{API_URL}{path}", json=payload, headers=headers, timeout=TIMEOUT)

def api_get(path: str, params: dict | None = None, headers: dict | None = None):
    return requests.get(f"{API_URL}{path}", params=params, headers=headers, timeout=TIMEOUT)

def api_post_multipart(path: str, data: dict, files: list, headers: dict | None = None):
    # files format: [("files", (filename, bytes, mime)), ...]
    return requests.post(f"{API_URL}{path}", data=data, files=files, headers=headers, timeout=TIMEOUT)

def admin_headers():
    if st.session_state.admin_token:
        return {"X-Admin-Token": st.session_state.admin_token}
    return {}


# -----------------------------
# Data loaders
# -----------------------------
def load_topics():
    r = api_get("/topics", {"user_id": st.session_state.user_id})
    r.raise_for_status()
    st.session_state.topics = r.json().get("topics", [])

    if st.session_state.topic_id is None and st.session_state.topics:
        general = next((t for t in st.session_state.topics if t["name"].lower() == "general"), None)
        st.session_state.topic_id = general["topic_id"] if general else st.session_state.topics[0]["topic_id"]

def load_history():
    if st.session_state.topic_id is None:
        st.session_state.messages = []
        return
    r = api_post("/get_history_topic", {"user_id": st.session_state.user_id, "topic_id": st.session_state.topic_id})
    r.raise_for_status()
    st.session_state.messages = r.json().get("history", [])

def topic_label_map():
    return {f'{t["name"]} ({t["role"]})': t["topic_id"] for t in st.session_state.topics}

def get_topic_by_id(tid: int):
    for t in st.session_state.topics:
        if int(t["topic_id"]) == int(tid):
            return t
    return None


# -----------------------------
# Landing (Admin / User)
# -----------------------------
if st.session_state.mode is None:
    st.subheader("Choose role")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Admin", use_container_width=True):
            st.session_state.mode = "admin"
            st.rerun()
    with c2:
        if st.button("Employee / User", use_container_width=True):
            st.session_state.mode = "user"
            st.rerun()
    st.stop()


# ============================================================
# ADMIN MODE
# ============================================================
if st.session_state.mode == "admin":
    st.sidebar.header("Admin")

    # ---- Step 1: check if any admin exists ----
    try:
        has_admin_resp = api_get("/admin/has_admin")
        has_admin_resp.raise_for_status()
        has_admin = bool(has_admin_resp.json().get("has_admin"))
    except Exception as e:
        st.error(f"Backend not ready (/admin/has_admin failed): {e}")
        st.stop()

    # ---- Step 2: if not logged in, show bootstrap OR login ----
    if not st.session_state.admin_token:
        if not has_admin:
            st.subheader("Create First Admin (one-time setup)")
            with st.form("bootstrap_admin"):
                u = st.text_input("Admin username")
                p = st.text_input("Admin password", type="password")
                p2 = st.text_input("Confirm password", type="password")
                ok = st.form_submit_button("Create First Admin")

            if ok:
                if not u.strip() or not p.strip():
                    st.warning("Username and password required.")
                elif p != p2:
                    st.warning("Passwords do not match.")
                else:
                    r = api_post("/admin/bootstrap", {"username": u.strip(), "password": p.strip()})
                    if r.status_code != 200:
                        st.error(f"Bootstrap failed: {r.text}")
                    else:
                        data = r.json()
                        st.session_state.admin_token = data["token"]
                        st.session_state.admin_username = data.get("admin_username") or u.strip()
                        st.success("First admin created and logged in.")
                        st.rerun()

            if st.button("Back"):
                st.session_state.mode = None
                st.rerun()

            st.stop()

        else:
            st.subheader("Admin Login")
            with st.form("admin_login"):
                u = st.text_input("Admin username")
                p = st.text_input("Admin password", type="password")
                ok = st.form_submit_button("Login")

            if ok:
                if not u.strip() or not p.strip():
                    st.warning("Username and password required.")
                else:
                    r = api_post("/admin/login", {"username": u.strip(), "password": p.strip()})
                    if r.status_code != 200:
                        st.error(f"Login failed: {r.text}")
                    else:
                        data = r.json()
                        st.session_state.admin_token = data["token"]
                        st.session_state.admin_username = data.get("admin_username") or u.strip()
                        st.success("Admin login successful.")
                        st.rerun()

            if st.button("Back"):
                st.session_state.mode = None
                st.rerun()

            st.stop()

    # ---- Admin is logged in ----
    with st.sidebar:
        st.write(f"✅ Logged in as **{st.session_state.admin_username}**")
        if st.button("Logout Admin"):
            try:
                api_post("/admin/logout", {}, headers=admin_headers())
            except Exception:
                pass
            st.session_state.admin_token = None
            st.session_state.admin_username = None
            st.rerun()

        st.divider()
        st.caption("Admin uses a backend user_id to create topics/upload docs.")
        if st.session_state.user_id is None:
            st.info("Set Admin backend username below.")

    # ---- Create/get a backend user_id for admin operations ----
    if st.session_state.user_id is None:
        st.subheader("Admin Identity (backend user_id)")
        with st.form("admin_identity"):
            username = st.text_input("Username for system (e.g., admin)", value=st.session_state.admin_username or "admin")
            ok = st.form_submit_button("Continue")

        if ok:
            r = api_post("/get_or_create_user", {"username": username.strip()})
            if r.status_code != 200:
                st.error(f"Failed creating user: {r.text}")
            else:
                data = r.json()
                st.session_state.user_id = data["user_id"]
                st.session_state.username = data["username"]
                load_topics()
                st.success("Admin identity set.")
                st.rerun()

        st.stop()

    # ---- Admin dashboard ----
    st.header("Admin Dashboard")

    if not st.session_state.topics:
        try:
            load_topics()
        except Exception as e:
            st.error(f"Failed to load topics: {e}")

    colA, colB = st.columns([2, 1], gap="large")

    # =========================
    # ✅ Create Topic + Improve Prompt + Choose Draft/Improved
    # =========================
    with colA:
        st.subheader("1) Create Topic")

        st.session_state.topic_name_draft = st.text_input(
            "Topic name",
            value=st.session_state.topic_name_draft,
            placeholder="e.g., Proteins and DNA / Company Policy / HR Docs",
            key="topic_name_input",
        )

        # draft input (admin typed)
        st.session_state.topic_prompt_draft = st.text_area(
            "Topic behavior prompt (your draft)",
            value=st.session_state.topic_prompt_draft,
            height=160,
            placeholder="e.g., You are an HR policy assistant. Only answer using uploaded documents...",
            key="topic_prompt_input",
        )

        # Improve / Create buttons
        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button("✨ Improve Prompt", use_container_width=True):
                if not st.session_state.topic_name_draft.strip():
                    st.warning("Enter a Topic name first.")
                else:
                    try:
                        r = api_post(
                            "/topics/improve_prompt",
                            {
                                "topic_name": st.session_state.topic_name_draft.strip(),
                                "draft_prompt": st.session_state.topic_prompt_draft.strip(),
                            },
                            headers=admin_headers(),
                        )
                        if r.status_code != 200:
                            st.error(f"Improve prompt failed: {r.text}")
                        else:
                            improved = r.json().get("improved_prompt", "").strip()
                            if not improved:
                                st.error("Backend returned empty improved prompt.")
                            else:
                                st.session_state.improved_prompt = improved
                                st.session_state.prompt_choice = "Improved"  # auto-select improved
                                st.success("Improved prompt generated. Choose Draft vs Improved below, then Create Topic.")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Improve prompt request failed: {e}")

        with btn2:
            if st.button("Create Topic", use_container_width=True):
                name = st.session_state.topic_name_draft.strip()
                draft_prompt = st.session_state.topic_prompt_draft.strip()
                improved_prompt = (st.session_state.improved_prompt or "").strip()

                if not name:
                    st.warning("Topic name required.")
                else:
                    final_prompt = draft_prompt
                    if st.session_state.prompt_choice == "Improved" and improved_prompt:
                        final_prompt = improved_prompt

                    r = api_post(
                        "/topics",
                        {"user_id": st.session_state.user_id, "name": name, "system_prompt": final_prompt},
                        headers=admin_headers(),
                    )
                    if r.status_code != 200:
                        st.error(f"Create topic failed: {r.text}")
                    else:
                        st.success("Topic created.")
                        # reset drafts
                        st.session_state.topic_name_draft = ""
                        st.session_state.topic_prompt_draft = ""
                        st.session_state.improved_prompt = ""
                        st.session_state.prompt_choice = "Draft"
                        load_topics()
                        st.rerun()

        # Chooser + previews (only if improved exists)
        if (st.session_state.improved_prompt or "").strip():
            st.radio(
                "Which prompt should be used for this topic?",
                ["Draft", "Improved"],
                key="prompt_choice",
                horizontal=True
            )
            with st.expander("Preview: Draft prompt", expanded=False):
                st.write(st.session_state.topic_prompt_draft)
            with st.expander("Preview: Improved prompt", expanded=True):
                st.write(st.session_state.improved_prompt)

        st.subheader("2) Select Topic")
        if st.session_state.topics:
            options = topic_label_map()
            labels = list(options.keys())
            current = None
            for label, tid in options.items():
                if tid == st.session_state.topic_id:
                    current = label
                    break

            selected = st.selectbox(
                "Topic",
                labels,
                index=labels.index(current) if current else 0,
                key="topic_select_admin",
            )
            st.session_state.topic_id = options[selected]
        else:
            st.info("No topics yet. Create one above.")

        st.subheader("3) Upload Documents to Selected Topic")
        selected_topic = get_topic_by_id(st.session_state.topic_id) if st.session_state.topic_id else None

        if selected_topic and selected_topic["name"].lower() == "general":
            st.warning("Uploading to General is disabled. Select a non-General topic.")
        elif selected_topic:
            files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True, key="uploader_admin")
            if st.button("Upload to Topic", key="upload_btn_admin"):
                if not files:
                    st.warning("Select at least one file.")
                else:
                    form_data = {"user_id": str(st.session_state.user_id)}
                    multipart_files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
                    resp = api_post_multipart(
                        f"/topics/{st.session_state.topic_id}/upload",
                        data=form_data,
                        files=multipart_files,
                        headers=admin_headers(),
                    )
                    if resp.status_code != 200:
                        st.error(f"Upload failed: {resp.status_code} {resp.text}")
                    else:
                        out = resp.json()
                        st.success(f"Uploaded. chunks={out.get('chunks')} namespace={out.get('namespace')}")
        else:
            st.info("Select a topic first.")

        st.subheader("4) Add Employee/User to Topic")
        if selected_topic and selected_topic["name"].lower() != "general":
            with st.form("add_member"):
                member_user = st.text_input("Employee username")
                role = st.selectbox("Role", ["employee", "admin"], index=0)
                ok_member = st.form_submit_button("Add/Update Member")

            if ok_member:
                if not member_user.strip():
                    st.warning("Username required.")
                else:
                    u = api_post("/get_or_create_user", {"username": member_user.strip()})
                    if u.status_code != 200:
                        st.error(f"Could not create user: {u.text}")
                    else:
                        r = api_post(
                            f"/topics/{st.session_state.topic_id}/add_member",
                            {"admin_user_id": st.session_state.user_id, "username": member_user.strip(), "role": role},
                            headers=admin_headers(),
                        )
                        if r.status_code != 200:
                            st.error(f"Add member failed: {r.text}")
                        else:
                            st.success("Member added/updated.")
        else:
            st.info("Select a non-General topic to add members.")

    # Right column
    with colB:
        st.subheader("Create Another Admin")
        st.caption("Only works when you are logged in as admin.")
        with st.form("create_admin"):
            new_u = st.text_input("New admin username")
            new_p = st.text_input("New admin password", type="password")
            ok_admin = st.form_submit_button("Create Admin")

        if ok_admin:
            if not new_u.strip() or not new_p.strip():
                st.warning("Username and password required.")
            else:
                r = api_post(
                    "/admin/create_admin",
                    {"username": new_u.strip(), "password": new_p.strip()},
                    headers=admin_headers(),
                )
                if r.status_code != 200:
                    st.error(f"Create admin failed: {r.text}")
                else:
                    st.success("New admin created.")

        st.divider()
        st.subheader("Switch to User Mode")
        if st.button("Go to User Mode", use_container_width=True):
            st.session_state.mode = "user"
            st.session_state.messages = []
            st.rerun()


# ============================================================
# USER MODE
# ============================================================
if st.session_state.mode == "user":
    if st.session_state.user_id is None:
        st.subheader("User Login")
        with st.form("user_login"):
            username = st.text_input("Enter your username")
            ok = st.form_submit_button("Login / Signup")

        if ok:
            if not username.strip():
                st.warning("Please enter a username.")
            else:
                r = api_post("/get_or_create_user", {"username": username.strip()})
                if r.status_code != 200:
                    st.error(f"Login failed: {r.text}")
                else:
                    data = r.json()
                    st.session_state.user_id = data["user_id"]
                    st.session_state.username = data["username"]
                    load_topics()
                    load_history()
                    st.rerun()

        if st.button("Back"):
            st.session_state.mode = None
            st.rerun()

        st.stop()

    with st.sidebar:
        st.subheader(f"Logged in as: {st.session_state.username}")

        if st.button("Reload topics"):
            load_topics()
            st.rerun()

        if not st.session_state.topics:
            load_topics()

        options = topic_label_map()
        if options:
            labels = list(options.keys())
            current = None
            for label, tid in options.items():
                if tid == st.session_state.topic_id:
                    current = label
                    break

            selected = st.selectbox(
                "Select topic",
                labels,
                index=labels.index(current) if current else 0,
                key="topic_select_user",
            )
            selected_tid = options[selected]
            if selected_tid != st.session_state.topic_id:
                st.session_state.topic_id = selected_tid
                load_history()
                st.rerun()

        if st.button("Refresh history"):
            load_history()
            st.rerun()

        if st.button("Logout"):
            st.session_state.mode = None
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.messages = []
            st.session_state.topics = []
            st.session_state.topic_id = None
            st.rerun()

    st.write("Pick a topic and chat. You can do small talk, RAG questions, or memory questions (e.g., “what was my first question?”).")

    for chat in st.session_state.messages:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    prompt = st.chat_input("Ask me anything...")
    if prompt:
        prompt = prompt.strip()
        st.session_state.messages.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                payload = {"user_id": st.session_state.user_id, "topic_id": st.session_state.topic_id, "text": prompt}
                r = api_post("/query", payload)
                if r.status_code != 200:
                    st.error(f"Backend error {r.status_code}: {r.text}")
                else:
                    data = r.json()
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    st.markdown(answer)

                    if citations:
                        with st.expander("Citations"):
                            for c in citations:
                                if isinstance(c, dict):
                                    st.write(f"- {c.get('source')} (page: {c.get('page')})")

                    st.session_state.messages.append({"role": "ai", "content": answer})