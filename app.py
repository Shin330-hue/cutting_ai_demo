import streamlit as st
import pandas as pd

# CSV読み込み
@st.cache_data
def load_knowledge():
    df = pd.read_csv("data//knowledge.csv")
    return df

def main():
    st.title("切削加工ナレッジ AI（簡易版）")

    # CSVをDataFrame化
    knowledge_df = load_knowledge()

    # 検索ワード入力欄
    query = st.text_input("キーワードを入力（例：びびり, 工具寿命など）", "")

    if st.button("検索"):
        if query.strip():
            # 簡易的に「キーワード列に含まれるかどうか」でフィルタ
            # 本来はベクトル検索等を行うことが多い
            filtered = knowledge_df[knowledge_df["キーワード"].str.contains(query)]
            if len(filtered) == 0:
                st.warning("該当なし")
            else:
                for idx, row in filtered.iterrows():
                    st.write(f"**現象**: {row['現象']}")
                    st.write(f"**原因**: {row['原因']}")
                    st.write(f"**対策**: {row['対策']}")
                    st.write("---")
        else:
            st.info("キーワードを入力してください")

if __name__ == "__main__":
    main()
