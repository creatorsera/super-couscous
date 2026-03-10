import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# ------------------------------------------------------------
# 1. Load the sentence transformer model (cached for performance)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the 'all-mpnet-base-v2' model – better accuracy, slower but fine for hundreds of keywords."""
    return SentenceTransformer('all-mpnet-base-v2')

# ------------------------------------------------------------
# 2. Scoring function: convert cosine similarity to 0‑10 scale
# ------------------------------------------------------------
def compute_scores(niche, keywords, model):
    """Return raw similarities and scaled scores (0‑10)."""
    niche_emb = model.encode([niche])
    keyword_embs = model.encode(keywords)
    similarities = cosine_similarity(niche_emb, keyword_embs)[0]

    def scale(sim):
        # Map similarity in [-1,1] to 0‑10:
        # - below 0 -> 0
        # - above 0.8 -> 10
        # - linear in between
        if sim < 0:
            return 0.0
        elif sim > 0.8:
            return 10.0
        else:
            return sim * 10 / 0.8   # 0 to 0.8 maps to 0‑10

    scores = [scale(s) for s in similarities]
    return similarities, scores

# ------------------------------------------------------------
# 3. Categorisation and reasoning
# ------------------------------------------------------------
def categorize(score):
    if score >= 7:
        return "Relevant"
    elif score >= 4:
        return "Maybe"
    else:
        return "Not Relevant"

def generate_reasoning(keyword, niche, score, category):
    """Create a short, meaningful explanation based on the score."""
    niche_preview = niche[:60] + "..." if len(niche) > 60 else niche
    if category == "Relevant":
        return f"Strong semantic match with your niche '{niche_preview}'. Score: {score:.1f}/10."
    elif category == "Maybe":
        return f"Partial match with your niche '{niche_preview}'. Score: {score:.1f}/10. May need manual review."
    else:
        return f"Weak semantic match with your niche '{niche_preview}'. Score: {score:.1f}/10. Likely irrelevant."

# ------------------------------------------------------------
# 4. Streamlit UI
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="Niche Keyword Relevance Checker", layout="wide")
    st.title("🔍 Niche Keyword Relevance Checker")
    st.markdown("Enter your niche description and a list of keywords to see how relevant each keyword is to your niche.")

    with st.expander("How it works"):
        st.write("""
        This tool uses a state-of-the-art sentence transformer model (`all-mpnet-base-v2`) to convert your niche description and each keyword into numerical embeddings (vectors). 
        It then computes the cosine similarity between the niche embedding and each keyword embedding. 
        The similarity score (ranging from -1 to 1) is scaled to a 0‑10 relevance score:
        - **0‑3.9**: Not Relevant
        - **4‑6.9**: Maybe Relevant
        - **7‑10**: Relevant
        The reasoning is generated based on the score.
        """)

    col1, col2 = st.columns(2)
    with col1:
        niche = st.text_area("📝 Describe your niche", height=150,
                             placeholder="e.g., I run a blog about vegan recipes and plant-based nutrition...")
    with col2:
        keywords_input = st.text_area("🔑 Enter keywords (one per line)", height=150,
                                      placeholder="vegan protein\ntofu recipes\nleather shoes\ngasoline car")

    if st.button("🚀 Check Relevance", type="primary"):
        if not niche.strip():
            st.error("Please enter a niche description.")
            return
        if not keywords_input.strip():
            st.error("Please enter at least one keyword.")
            return

        keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]
        if not keywords:
            st.error("No valid keywords found.")
            return

        # Load model (first run downloads the model – may take a moment)
        with st.spinner("Loading AI model... (this may take a moment on first run)"):
            model = load_model()

        # Compute scores with progress feedback
        progress_bar = st.progress(0, text="Computing relevance...")
        similarities, scores = compute_scores(niche, keywords, model)
        progress_bar.progress(100)

        # Build result table
        categories = [categorize(s) for s in scores]
        reasonings = [generate_reasoning(kw, niche, scores[i], categories[i]) for i, kw in enumerate(keywords)]

        df = pd.DataFrame({
            "Keyword": keywords,
            "Score (0-10)": [round(s, 2) for s in scores],
            "Relevance": categories,
            "Reasoning": reasonings
        })

        # 5. Display summary chart
        st.subheader("📊 Relevance Summary")
        chart_data = df["Relevance"].value_counts().reset_index()
        chart_data.columns = ["Relevance", "Count"]
        fig = px.pie(chart_data, values="Count", names="Relevance",
                     title="Keyword Relevance Distribution",
                     color="Relevance",
                     color_discrete_map={"Relevant": "green", "Maybe": "orange", "Not Relevant": "red"})
        st.plotly_chart(fig, use_container_width=True)

        # 6. Display detailed results
        st.subheader("📋 Detailed Results")
        st.dataframe(df, use_container_width=True)

        # 7. Download and copy buttons
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download as CSV",
                               data=csv,
                               file_name="keyword_relevance.csv",
                               mime="text/csv")
        with col2:
            # Provide a tab-separated version for easy copying
            copy_text = df.to_csv(index=False, sep="\t")
            st.text_area("📋 Copy all (tab‑separated)", copy_text, height=200)

        # Optional: show raw similarities for debugging
        with st.expander("Show raw cosine similarities"):
            st.write(dict(zip(keywords, similarities)))

if __name__ == "__main__":
    main()
