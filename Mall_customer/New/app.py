import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from pymongo import MongoClient
import bcrypt
import datetime

# ==== MongoDB Setup ====
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_segmentation_db"]
users_col = db["users"]
results_col = db["cluster_results"]

# ==== Authentication Functions ====
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

def create_user(email, username, password):
    if users_col.find_one({"_id": email}):
        return False
    users_col.insert_one({
        "_id": email,
        "username": username,
        "password": hash_password(password)
    })
    return True

def login_user(email, password):
    user = users_col.find_one({"_id": email})
    if user and check_password(password, user["password"]):
        return user
    return None

# ==== Streamlit Setup ====
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🛍️ Customer Segmentation App")

# === Theme Toggle ===
theme = st.sidebar.radio("🌙 Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """<style>
        .main { background-color: #0e1117; color: white; }
        </style>""",
        unsafe_allow_html=True
    )

menu = ["Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)

if "user" not in st.session_state:
    st.session_state.user = None

# ==== Login / Signup ====
if st.session_state.user is None:
    if choice == "Login":
        st.subheader("🔐 Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")
        if login_btn:
            user = login_user(email, password)
            if user:
                st.success(f"Welcome, {user['username']}!")
                st.session_state.user = user
                st.query_params.clear()
                st.rerun()
            else:
                st.error("Invalid credentials")

    elif choice == "Sign Up":
        st.subheader("📝 Create Account")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        signup_btn = st.button("Sign Up")
        if signup_btn:
            if create_user(email, username, password):
                st.success("Account created! Please log in.")
            else:
                st.error("Account already exists.")

# ==== Main App ====
if st.session_state.user:
    st.sidebar.success(f"Logged in as: {st.session_state.user['username']}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.query_params.clear()
        st.rerun()

    @st.cache_data
    def load_data():
        return pd.read_csv("Mall_Customers.csv")

    df = load_data()
    st.subheader("Customer Data")
    st.dataframe(df)

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    st.sidebar.subheader("🔧 Clustering Settings")
    show_elbow = st.sidebar.checkbox("Show Elbow Plot")
    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 5)

    if show_elbow:
        st.subheader("📈 Elbow Plot")
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42)
            km.fit(X)
            wcss.append(km.inertia_)
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, 11), wcss, marker='o')
        ax1.set_title("The Elbow Method")
        ax1.set_xlabel("Clusters")
        ax1.set_ylabel("WCSS")
        st.pyplot(fig1)

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X)
    df['Cluster'] = clusters

    # === Label Your Clusters ===
    st.subheader("📝 Label Your Clusters")
    cluster_labels = []
    for i in range(num_clusters):
        label = st.text_input(f"Label for Cluster {i+1}", value=f"Cluster {i+1}")
        cluster_labels.append(label)

    df['Cluster_Label'] = df['Cluster'].apply(lambda x: cluster_labels[x])

    # Cluster Plot
    st.subheader("🧩 Segmentation Result")
    fig2, ax2 = plt.subplots()
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'olive', 'pink']
    for i in range(num_clusters):
        ax2.scatter(X.values[clusters == i, 0], X.values[clusters == i, 1],
                    s=100, c=colors[i], label=cluster_labels[i])
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                s=300, c='yellow', marker='X', label='Centroids')
    ax2.set_title("Customer Segments")
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score (1-100)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("📄 Clustered Data")
    st.dataframe(df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',
                     'Spending Score (1-100)', 'Cluster', 'Cluster_Label']])

    # === Export Buttons ===
    st.download_button("📥 Download as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="clustered_customers.csv", mime="text/csv")

    st.download_button("📥 Download as JSON", df.to_json(orient="records").encode("utf-8"),
                       file_name="clustered_customers.json", mime="application/json")

    # === Optional 3D Cluster Plot ===
    if st.checkbox("🧠 Show 3D Cluster Plot"):
        fig3d = px.scatter_3d(
            df, x="Age", y="Annual Income (k$)", z="Spending Score (1-100)",
            color="Cluster_Label", symbol="Gender",
            title="3D Customer Clusters"
        )
        st.plotly_chart(fig3d)

    # === Save to MongoDB ===
    if st.button("💾 Save Results"):
        try:
            data_to_save = {
                "user_id": st.session_state.user["_id"],
                "num_clusters": int(num_clusters),
                "clustered_data": df.astype(object).to_dict(orient="records"),
                "timestamp": datetime.datetime.now()
            }
            result = results_col.insert_one(data_to_save)
            if result.inserted_id:
                st.success("✅ Clustering result saved to MongoDB!")
            else:
                st.error("❌ Failed to save result.")
        except Exception as e:
            st.error(f"MongoDB Error: {e}")

    # === View History ===
    if st.checkbox("📜 View Past Cluster History"):
        results = results_col.find({"user_id": st.session_state.user["_id"]}).sort("timestamp", -1)
        count = 0
        for r in results:
            count += 1
            st.markdown(f"#### Saved on: {r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            df_old = pd.DataFrame(r["clustered_data"])
            st.dataframe(df_old.head())
        if count == 0:
            st.info("No past clustering history found.")
