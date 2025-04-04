# pip install google-generativeai

import google.generativeai as genai
import os

# Gemini API設定
API_KEY = "AIzaSyBzjQUxlNlpMQEdvYKqLDoL94XUN7YXxb8"  # 環境変数から取得する方が安全です
genai.configure(api_key=API_KEY)

# モデル設定
model = genai.GenerativeModel('gemini-pro')

@st.cache_data
def get_llm_response(query, context_data):
    # プロンプトテンプレート
    prompt = f"""
    あなたは切削加工のプロフェッショナルです。以下の類似事例を参考に、ユーザーの質問に具体的に答えてください。
    回答は簡潔に、現場で実践できる内容を心がけてください。
    
    【参考情報】
    {context_data}
    
    【質問】
    {query}
    
    【回答】
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"APIエラー: {e}")
        return "申し訳ありません。一時的なエラーが発生しました。"

def main():
    # 既存コードは同じ...
    
    # 検索処理
    if search_button or ('query' in st.session_state and st.session_state.query != ""):
        if query.strip():
            # ベクトル検索処理
            # ...
            
            # 上位の検索結果をLLMへの文脈として使用
            context_data = ""
            for idx in top_indices[:2]:  # 上位2件を文脈として使用
                if similarities[idx] > 0.3:
                    context_data += f"事例: {knowledge_df.iloc[idx]['現象']}\n"
                    context_data += f"原因: {knowledge_df.iloc[idx]['原因']}\n"
                    context_data += f"対策: {knowledge_df.iloc[idx]['対策']}\n\n"
            
            # LLMで回答生成（検索結果が十分あるとき）
            if context_data:
                with st.spinner("AI回答を生成中..."):
                    llm_response = get_llm_response(query, context_data)
                    
                    # LLM回答を表示
                    st.markdown("## AIアドバイス")
                    st.markdown(llm_response)
                    st.caption("※この回答は参考情報です。最終判断は専門家に相談してください。")
                    
            # 以下、検索結果の表示は同じ...