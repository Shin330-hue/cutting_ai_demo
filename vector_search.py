# 必要なライブラリをインストール
# pip install sentence-transformers scikit-learn streamlit pandas

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# モデルとデータのロード
@st.cache_resource
def load_model():
    return SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens')

@st.cache_data
def load_knowledge():
    df = pd.read_csv("data/knowledge.csv")
    return df

# ベクトル計算（事前計算）
@st.cache_data
def compute_embeddings(df, model):
    # 現象と原因を組み合わせてベクトル化
    texts = df['現象'] + " " + df['原因']
    embeddings = model.encode(texts)
    return embeddings

def main():
    st.title("切削加工ナレッジ AI（ベクトル検索版）")

    # モデルとデータをロード
    model = load_model()
    knowledge_df = load_knowledge()
    knowledge_embeddings = compute_embeddings(knowledge_df, model)

    # 検索クエリ入力欄
    query = st.text_input("質問を入力（例：工具がすぐに折れる、仕上げ面が粗いなど）", "")

    if st.button("検索"):
        if query.strip():
            # クエリをベクトル化
            query_embedding = model.encode([query])
            
            # コサイン類似度計算
            similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
            
            # 上位3件の類似データを取得
            top_indices = np.argsort(similarities)[::-1][:3]
            
            # 結果表示
            for i, idx in enumerate(top_indices):
                similarity = similarities[idx]
                if similarity > 0.3:  # 類似度しきい値
                    st.write(f"**検索結果 {i+1}** (類似度: {similarity:.2f})")
                    st.write(f"**現象**: {knowledge_df.iloc[idx]['現象']}")
                    st.write(f"**原因**: {knowledge_df.iloc[idx]['原因']}")
                    st.write(f"**対策**: {knowledge_df.iloc[idx]['対策']}")
                    st.write("---")
                else:
                    if i == 0:
                        st.warning("十分に関連する情報が見つかりませんでした")
                    break
        else:
            st.info("質問を入力してください")

if __name__ == "__main__":
    main()

# 上記コードに追加・変更

def main():
    st.set_page_config(page_title="切削加工アドバイザー", layout="wide")
    
    st.title("切削加工ナレッジ AI")
    st.markdown("#### 現場の疑問をAIが解決します")
    
    # サイドバーにカテゴリフィルター
    st.sidebar.header("検索オプション")
    categories = ["すべて"] + list(knowledge_df["キーワード"].str.split(",").explode().unique())
    selected_category = st.sidebar.selectbox("カテゴリで絞り込み", categories)
    
    # よく検索されるキーワード（仮のデータ、後で実際の検索履歴から生成する）
    common_queries = ["びびり振動", "工具寿命", "表面粗さ", "切りくず"]
    st.sidebar.header("よく調べられる内容")
    for query in common_queries:
        if st.sidebar.button(query):
            st.session_state.query = query  # クリックしたキーワードをセット
    
    # メイン検索エリア
    col1, col2 = st.columns([3, 1])
    with col1:
        # セッション状態があれば利用
        if 'query' not in st.session_state:
            st.session_state.query = ""
        
        query = st.text_input("質問を入力（例：工具がすぐに折れる、仕上げ面が粗いなど）", st.session_state.query)
    
    with col2:
        search_button = st.button("検索", use_container_width=True)
    
    # 検索処理
    if search_button or ('query' in st.session_state and st.session_state.query != ""):
        if query.strip():
            # ベクトル検索処理（前述のコードと同じ）
            # ...
            
            # 結果をカードUIで表示
            st.subheader("検索結果")
            for i, idx in enumerate(top_indices):
                if similarities[idx] > 0.3:
                    with st.expander(f"{knowledge_df.iloc[idx]['現象']} (類似度: {similarities[idx]:.2f})", expanded=True if i==0 else False):
                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown("### 現象")
                            st.write(knowledge_df.iloc[idx]['現象'])
                        with cols[1]:
                            st.markdown("### 原因")
                            st.write(knowledge_df.iloc[idx]['原因'])
                        with cols[2]:
                            st.markdown("### 対策")
                            st.write(knowledge_df.iloc[idx]['対策'])
                        
                        # フィードバックボタン
                        feedback_cols = st.columns([1,1,4])
                        with feedback_cols[0]:
                            st.button("👍 役立った", key=f"useful_{idx}")
                        with feedback_cols[1]:
                            st.button("👎 役立たなかった", key=f"not_useful_{idx}")