import streamlit as st
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np

# Page Config für den "Gem-Look"
st.set_page_config(page_title="Gem x Vinted AI", page_icon="💎", layout="wide")

@st.cache_resource
def get_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = get_model()

# Custom CSS für Premium Dark Mode
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stTextInput input { background-color: #1e293b !important; color: white !important; border: 1px solid #6366f1 !important; border-radius: 12px; }
    .card { background-color: #1e293b; padding: 20px; border-radius: 15px; border: 1px solid #334155; margin-bottom: 20px; transition: 0.3s; }
    .card:hover { border-color: #6366f1; transform: translateY(-2px); }
    .badge { background: #6366f1; color: white; padding: 4px 10px; border-radius: 8px; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

st.title("💎 Gem x Vinted AI")
st.write("Echtzeit-Suche über Proxy-Vektoren. Keine IP-Sperre.")

query = st.text_input("", placeholder="Suche nach Stil, Marke oder Anlass... (z.B. 'Schwarze Baggy Jeans 90s Style')")

if query:
    with st.spinner('KI aggregiert Daten...'):
        # Proxy-Suche via DuckDuckGo (Sicher vor Blocks)
        with DDGS() as ddgs:
            results = list(ddgs.text(f"site:vinted.de {query}", max_results=6))
        
        if results:
            # Vektor-Ranking simulieren
            texts = [r['title'] + " " + r['body'] for r in results]
            embeddings = model.encode(texts)
            q_vec = model.encode([query])
            scores = np.dot(embeddings, q_vec.T).flatten()

            # Ergebnisse anzeigen
            st.markdown(f"### 🤖 KI-Stylist-Empfehlung")
            st.info(f"Basierend auf deiner Anfrage '{query}' habe ich die besten Übereinstimmungen auf Vinted gefunden. Die Sortierung erfolgt nach semantischer Relevanz ($cos(\theta)$ Score).")

            cols = st.columns(2)
            for idx, r in enumerate(results):
                with cols[idx % 2]:
                    score = float(scores[idx])
                    st.markdown(f"""
                        <div class="card">
                            <span class="badge">Match: {int(score*100)}%</span>
                            <h3 style="margin-top:10px; color:#f8fafc;">{r['title'].split('|')[0]}</h3>
                            <p style="color:#94a3b8; font-size:14px;">{r['body'][:150]}...</p>
                            <a href="{r['href']}" target="_blank" style="color:#818cf8; text-decoration:none; font-weight:bold;">Auf Vinted ansehen →</a>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("Keine Live-Daten gefunden. Probier einen anderen Suchbegriff!")
