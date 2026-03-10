import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import io

# ---------- Page Config ----------
st.set_page_config(
    page_title="Keyword Relevance Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS for a modern look ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .keyword-count {
        font-weight: bold;
        color: #FF5722;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎯 Keyword Relevance Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Semantic relevance checker for your niche – powered by state‑of‑the‑art AI</p>', unsafe_allow_html=True)

# ---------- Sidebar Settings ----------
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_options = {
        "all-MiniLM-L6-v2": "Fast (500 keywords/sec)",
        "all-mpnet-base-v2": "Accurate (150 keywords/sec)",
        "all-distilroberta-v1": "Balanced (200 keywords/sec)"
    }
    selected_model = st.selectbox(
        "Choose model",
        options=list(model_options.keys()),
        format_func=lambda x: f"{x} – {model_options[x]}"
    )
    
    # Category thresholds
    st.subheader("🎚️ Relevance thresholds")
    relevant_thresh = st.slider("Relevant (score ≥)", 5, 10, 7, 1)
    maybe_thresh = st.slider("Maybe (score ≥)", 0, relevant_thresh-1, 4, 1)
    
    # Advanced options
    with st.expander("Advanced"):
        show_raw_similarity = st.checkbox("Show raw cosine similarity", False)
        batch_size = st.number_input("Batch size (for large lists)", 32, 512, 128, 32)
        deduplicate = st.checkbox("Remove duplicate keywords", True)
        strip_whitespace = st.checkbox("Strip whitespace", True)
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Upload a CSV or TXT file for bulk checking.")
    st.markdown("📊 Results can be downloaded as CSV/Excel/JSON.")

# ---------- Main Area ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Your Niche")
    niche = st.text_area(
        "Describe your niche in detail (e.g., 'I run a blog about vegan recipes and plant-based nutrition')",
        height=150,
        key="niche_input"
    )
    
    # Demo button
    if st.button("✨ Load example", type="secondary"):
        niche = "Sustainable living tips, zero waste products, and eco-friendly lifestyle."
        st.session_state.niche_input = niche
        st.rerun()

with col2:
    st.subheader("🔑 Your Keywords")
    input_method = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)
    
    keywords = []
    if input_method == "Paste text":
        keywords_text = st.text_area(
            "Enter one keyword per line",
            height=150,
            placeholder="vegan protein\ntofu recipes\nleather shoes\ngasoline car"
        )
        if keywords_text:
            keywords = [k.strip() for k in keywords_text.split("\n") if k.strip()]
    else:
        uploaded_file = st.file_uploader("Choose a file (TXT or CSV)", type=["txt", "csv"])
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            keywords = [k.strip() for k in content.split("\n") if k.strip()]
    
    # Keyword stats
    if keywords:
        original_count = len(keywords)
        if deduplicate:
            keywords = list(dict.fromkeys(keywords))  # preserve order
        st.caption(f"📊 {len(keywords)} unique keywords" + (f" (from {original_count} total)" if original_count != len(keywords) else ""))

# ---------- Check Relevance Button ----------
check_button = st.button("🚀 Check Relevance", type="primary", use_container_width=True)

if check_button:
    if not niche.strip():
        st.error("Please enter a niche description.")
        st.stop()
    if not keywords:
        st.error("Please provide at least one keyword.")
        st.stop()
    
    # ---------- Load Model ----------
    @st.cache_resource(show_spinner=False)
    def load_model(model_name):
        with st.spinner(f"Loading {model_name}... (first run may take a minute)"):
            return SentenceTransformer(model_name)
    
    model = load_model(selected_model)
    
    # ---------- Compute Embeddings with Progress ----------
    st.subheader("⏳ Processing...")
    progress_bar = st.progress(0, text="Starting...")
    status_text = st.empty()
    
    # Encode niche once
    niche_emb = model.encode([niche])
    
    # Process keywords in batches
    batch_size = min(batch_size, len(keywords))
    all_similarities = []
    
    start_time = time.time()
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i+batch_size]
        batch_embs = model.encode(batch)
        sims = cosine_similarity(niche_emb, batch_embs)[0]
        all_similarities.extend(sims)
        
        # Update progress
        progress = min(1.0, (i + len(batch)) / len(keywords))
        elapsed = time.time() - start_time
        eta = (elapsed / (i+len(batch))) * (len(keywords) - (i+len(batch))) if i+len(batch) > 0 else 0
        progress_bar.progress(progress, text=f"Batch {i//batch_size+1}/{(len(keywords)-1)//batch_size+1} – ETA: {eta:.1f}s")
        status_text.text(f"Processed {min(i+batch_size, len(keywords))}/{len(keywords)} keywords")
    
    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Done! Processed {len(keywords)} keywords in {time.time()-start_time:.2f} seconds.")
    
    # ---------- Scale Similarities to 0-10 ----------
    def scale_similarity(sim):
        # Map similarity in [0,1] to 0-10 (since for relevant niches, sim is usually >0)
        # Using linear mapping from 0.2 (0) to 0.8 (10)
        if sim < 0.2:
            return 0.0
        elif sim > 0.8:
            return 10.0
        else:
            return (sim - 0.2) * 10 / 0.6
    
    scores = [scale_similarity(s) for s in all_similarities]
    
    # ---------- Categorise ----------
    def categorize(score):
        if score >= relevant_thresh:
            return "Relevant"
        elif score >= maybe_thresh:
            return "Maybe"
        else:
            return "Not Relevant"
    
    categories = [categorize(s) for s in scores]
    
    # ---------- Generate Reasoning ----------
    def generate_reasoning(keyword, niche, score, cat):
        preview = niche[:50] + "..." if len(niche) > 50 else niche
        if cat == "Relevant":
            return f"✅ Strong semantic match with '{preview}' – score {score:.1f}/10."
        elif cat == "Maybe":
            return f"⚠️ Partial match with '{preview}' – score {score:.1f}/10. Consider manual review."
        else:
            return f"❌ Weak match with '{preview}' – score {score:.1f}/10. Likely irrelevant."
    
    reasons = [generate_reasoning(kw, niche, scores[i], categories[i]) for i, kw in enumerate(keywords)]
    
    # ---------- Build DataFrame ----------
    df = pd.DataFrame({
        "Keyword": keywords,
        "Score (0-10)": [round(s, 2) for s in scores],
        "Relevance": categories,
        "Reasoning": reasons
    })
    if show_raw_similarity:
        df["Raw Cosine Sim"] = [round(s, 3) for s in all_similarities]
    
    # ---------- Visualisations ----------
    st.subheader("📊 Summary Dashboard")
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        # Pie chart
        counts = df["Relevance"].value_counts().reset_index()
        counts.columns = ["Relevance", "Count"]
        fig_pie = px.pie(
            counts,
            values="Count",
            names="Relevance",
            color="Relevance",
            color_discrete_map={"Relevant": "#4CAF50", "Maybe": "#FFC107", "Not Relevant": "#F44336"},
            title="Relevance Distribution"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        # Histogram of scores
        fig_hist = px.histogram(
            df,
            x="Score (0-10)",
            nbins=20,
            title="Score Distribution",
            color_discrete_sequence=["#2196F3"]
        )
        fig_hist.add_vline(x=relevant_thresh, line_dash="dash", line_color="green", annotation_text="Relevant")
        fig_hist.add_vline(x=maybe_thresh, line_dash="dash", line_color="orange", annotation_text="Maybe")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Optional word cloud (if wordcloud library is installed)
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        # Create a frequency dict: keyword -> score
        word_freq = dict(zip(df["Keyword"], df["Score (0-10)"]))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except ImportError:
        st.info("Install `wordcloud` to see a keyword cloud.")
    
    # ---------- Interactive Results Table ----------
    st.subheader("📋 Detailed Results")
    
    # Add color formatting via pandas styler
    def color_relevance(val):
        if val == "Relevant":
            return "background-color: #4CAF50; color: white"
        elif val == "Maybe":
            return "background-color: #FFC107; color: black"
        else:
            return "background-color: #F44336; color: white"
    
    styled_df = df.style.applymap(color_relevance, subset=["Relevance"])
    
    # Display as editable? For simplicity, we use static dataframe
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # ---------- Export Options ----------
    st.subheader("💾 Export Results")
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
    
    with export_col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSV", data=csv, file_name="relevance_results.csv", mime="text/csv")
    
    with export_col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        st.download_button("📥 Excel", data=excel_buffer.getvalue(), file_name="relevance_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    with export_col3:
        json_str = df.to_json(orient='records', indent=2)
        st.download_button("📥 JSON", data=json_str, file_name="relevance_results.json", mime="application/json")
    
    with export_col4:
        # Copy to clipboard using st.code (user can manually copy)
        st.code(df.to_csv(index=False, sep="\t"), language="text")
