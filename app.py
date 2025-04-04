# 必要なライブラリをインストール
# pip install sentence-transformers scikit-learn streamlit pandas google-generativeai python-dotenv

import streamlit as st
# 重要: set_page_configはインポート直後に呼び出す必要がある
st.set_page_config(page_title="切削加工アドバイザー", layout="wide")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# キャラクターごとのプロンプト設定
character_prompts = {
    "真面目": """
    あなたは切削加工のプロフェッショナルです。常に敬語を使い、正確で詳細な説明を心がけてください。
    専門用語を適切に使用し、論理的で筋道立てた説明をしてください。
    回答は簡潔かつ正確に、現場で実践できる内容を心がけてください。
    """,
    
    "フランク": """
    あなたは切削加工のプロフェッショナルです。くだけた言葉遣いで、わかりやすく簡潔に説明してください。
    「〜だよ」「〜だね」といったフレンドリーな口調を使い、専門用語は必要最低限に抑えて説明してください。
    話し言葉のようなカジュアルな言い回しを使って、親しみやすさを大切にしてください。
    回答は簡潔に、現場で実践できる内容を心がけてください。
    """,
    
    "お姉さん": """
    あなたは切削加工のプロフェッショナルであり、優しいお姉さん的存在です。
    「〜ね」「〜よ」といった女性的な口調で、優しく教えるように説明してください。
    時々「大丈夫よ」「頑張ってね」などの励ましの言葉も入れてください。
    専門的な内容も初心者にもわかりやすく噛み砕いて説明し、回答は現場で実践できる内容を心がけてください。
    """,
    
    "厳しい先輩": """
    あなたは切削加工のプロフェッショナルであり、厳しいが信頼できる先輩です。
    「〜だ」「〜しろ」といった少し命令口調で話し、時に厳しい指摘も躊躇わずにしてください。
    「甘いな」「基本だろ」などの厳しい言葉も交えつつ、本質的には役立つアドバイスを提供してください。
    回答は簡潔に、現場で実践できる内容を心がけてください。
    """,
    
    "関西弁": """
    あなたは切削加工のプロフェッショナルで、関西出身です。
    「〜やで」「〜やろ」「〜ちゃう？」などの関西弁で親しみやすく説明してください。
    「せやな」「ほんまに」「あかん」などの関西特有の表現も適度に使ってください。
    ユーモアも交えながら、回答は現場で実践できる内容を心がけてください。
    """
}

# モデルとデータのロード
@st.cache_resource
def load_model():
    return SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens')

@st.cache_data
def load_knowledge():
    try:
        df = pd.read_csv("data/knowledge.csv")
        return df
    except FileNotFoundError:
        # ファイルが見つからない場合は別のパスを試す
        try:
            df = pd.read_csv("knowledge.csv")
            return df
        except FileNotFoundError:
            st.error("knowledge.csvファイルが見つかりません")
            return pd.DataFrame(columns=['現象', '原因', '対策', 'キーワード'])

# ベクトル計算（事前計算）
@st.cache_data
def compute_embeddings(df, _model):
    # 現象と原因を組み合わせてベクトル化
    texts = df['現象'] + " " + df['原因']
    embeddings = _model.encode(texts)
    return embeddings

# Gemini API設定
API_KEY = os.environ.get('GEMINI_API_KEY')
gemini_model = None

if API_KEY:
    genai.configure(api_key=API_KEY)
    try:
        # デバッグ情報をサイドバーに表示（デプロイ後に確認できるように）
        st.sidebar.write("API_KEY設定: OK")
        
        # モデル取得を試みる - 複数のモデル名で試す
        model_names_to_try = [
            'models/gemini-1.5-pro',
            'gemini-1.5-pro',
            'models/gemini-pro',
            'gemini-pro',
            'models/gemini-1.0-pro',
            'gemini-1.0-pro'
        ]
        
        exception_messages = []
        for model_name in model_names_to_try:
            try:
                st.sidebar.write(f"モデル {model_name} を試しています...")
                gemini_model = genai.GenerativeModel(model_name)
                st.sidebar.success(f"モデル {model_name} の読み込み成功！")
                break
            except Exception as e:
                exception_messages.append(f"{model_name}: {str(e)}")
                continue
        
        if gemini_model is None:
            st.sidebar.error("すべてのモデル名で失敗しました")
            st.sidebar.error("\n".join(exception_messages))
                
    except Exception as e:
        st.sidebar.error(f"Gemini API接続エラー: {str(e)}")
else:
    st.sidebar.warning("⚠️ GEMINI_API_KEYが設定されていません。LLM機能は使用できません。")

# LLM回答生成
@st.cache_data
def get_llm_response(query, context_data, _model, character_type):
    if not API_KEY or _model is None:
        return "APIキーが設定されていないか、モデルの初期化に失敗したため、LLM機能は利用できません。"
    
    # キャラクター設定を取得（デフォルトは真面目）
    character_setting = character_prompts.get(character_type, character_prompts["真面目"])
    
    # プロンプトテンプレート
    prompt = f"""
    {character_setting}
    
    以下の類似事例を参考に、ユーザーの質問に具体的に答えてください。
    
    【参考情報】
    {context_data}
    
    【質問】
    {query}
    
    【回答】
    """
    
    try:
        # デバッグメッセージ
        st.sidebar.write(f"LLM APIを呼び出し中... ({character_type}キャラクター)")
        
        # セーフティ設定
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH",
            },
        ]
        
        # 応答生成を試みる
        response = _model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={"temperature": 0.7}  # キャラクター性を出すために少し温度を上げる
        )
        
        # デバッグメッセージ
        st.sidebar.success("LLM API呼び出し成功")
        
        # APIレスポンスの形式が変わっていることもあるので、柔軟に対応
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts'):
            return ''.join([part.text for part in response.parts])
        else:
            return str(response)
    except Exception as e:
        st.sidebar.error(f"LLM APIエラー: {str(e)}")
        return f"申し訳ありません。AIアドバイス生成中にエラーが発生しました: {str(e)}"

# フィードバックを記録する関数（将来的にDB連携などを想定）
def log_feedback(item_id, feedback_type, query, response):
    """
    フィードバックを記録する関数（現在はセッション内のみ）
    将来的にはCSV、DB、Googleスプレッドシートなどに記録することを想定
    """
    if 'feedback_history' not in st.session_state:
        st.session_state.feedback_history = []
    
    # フィードバック情報を記録
    feedback_data = {
        'item_id': item_id,
        'feedback': feedback_type,
        'query': query,
        'response': response,
        'timestamp': pd.Timestamp.now()
    }
    
    st.session_state.feedback_history.append(feedback_data)
    return len(st.session_state.feedback_history)

def main():
    st.title("切削加工ナレッジ AI")
    st.markdown("#### 現場の疑問をAIが解決します")
    
    # サイドバーにキャラクター選択を追加
    st.sidebar.header("AIキャラクター設定")
    character = st.sidebar.selectbox(
        "AIアドバイザーのキャラクターを選択",
        ["真面目", "フランク", "お姉さん", "厳しい先輩", "関西弁"]
    )
    
    # モデルとデータをロード
    model = load_model()
    knowledge_df = load_knowledge()
    
    # データが存在する場合のみエンベディングを計算
    if not knowledge_df.empty:
        knowledge_embeddings = compute_embeddings(knowledge_df, model)
    
        # サイドバーにカテゴリフィルター
        st.sidebar.header("検索オプション")
        categories = ["すべて"] + list(knowledge_df["キーワード"].str.split(",").explode().unique())
        selected_category = st.sidebar.selectbox("カテゴリで絞り込み", categories)
        
        # よく検索されるキーワード
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
                # クエリをベクトル化
                query_embedding = model.encode([query])
                
                # コサイン類似度計算
                similarities = cosine_similarity(query_embedding, knowledge_embeddings)[0]
                
                # カテゴリフィルター適用（「すべて」以外が選択されている場合）
                if selected_category != "すべて":
                    # カテゴリに一致する行のみフィルタリング
                    filtered_indices = knowledge_df["キーワード"].str.contains(selected_category)
                    # フィルタリングされていない行の類似度を0にする
                    for i in range(len(similarities)):
                        if not filtered_indices.iloc[i]:
                            similarities[i] = 0
                
                # 上位3件の類似データを取得
                top_indices = np.argsort(similarities)[::-1][:3]
                
                # LLM連携 - 上位の検索結果をLLMへの文脈として使用
                context_data = ""
                for idx in top_indices[:2]:  # 上位2件を文脈として使用
                    if similarities[idx] > 0.3:
                        context_data += f"事例: {knowledge_df.iloc[idx]['現象']}\n"
                        context_data += f"原因: {knowledge_df.iloc[idx]['原因']}\n"
                        context_data += f"対策: {knowledge_df.iloc[idx]['対策']}\n\n"
                
                # LLMで回答生成（検索結果が十分あるとき）
                if context_data and API_KEY and gemini_model is not None:
                    with st.spinner("AI回答を生成中..."):
                        llm_response = get_llm_response(query, context_data, gemini_model, character)
                        
                        # LLM回答を表示
                        st.markdown("## AIアドバイス")
                        st.markdown(llm_response)
                        st.caption(f"※この回答は {character} キャラクターによる回答です。最終判断は専門家に相談してください。")
                
                # 検索結果をカードUIで表示
                st.subheader("検索結果")
                found_results = False
                for i, idx in enumerate(top_indices):
                    if similarities[idx] > 0.3:  # 類似度しきい値
                        found_results = True
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
                                if st.button("👍 役立った", key=f"useful_{idx}"):
                                    count = log_feedback(idx, "useful", query, knowledge_df.iloc[idx]['対策'])
                                    st.success(f"フィードバックありがとうございます！(#{count})")
                            with feedback_cols[1]:
                                if st.button("👎 役立たなかった", key=f"not_useful_{idx}"):
                                    count = log_feedback(idx, "not_useful", query, knowledge_df.iloc[idx]['対策'])
                                    st.error(f"ご意見ありがとうございます。改善に努めます。(#{count})")
                
                if not found_results:
                    st.warning("十分に関連する情報が見つかりませんでした")
            else:
                st.info("質問を入力してください")
    else:
        st.error("ナレッジデータが読み込めません。CSVファイルを確認してください。")

    # フィードバック履歴の表示（開発・デバッグ用）
    if st.sidebar.checkbox("フィードバック履歴を表示", value=False):
        if 'feedback_history' in st.session_state and st.session_state.feedback_history:
            st.sidebar.write("### フィードバック履歴")
            for i, feedback in enumerate(st.session_state.feedback_history):
                st.sidebar.write(f"**#{i+1}** - {feedback['timestamp']}")
                st.sidebar.write(f"質問: {feedback['query']}")
                st.sidebar.write(f"評価: {feedback['feedback']}")
        else:
            st.sidebar.write("フィードバック履歴はまだありません")

if __name__ == "__main__":
    main()