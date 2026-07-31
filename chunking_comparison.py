import os
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

# 1. Đo độ dài token bằng tiktoken (chỉ để minh họa sự khác biệt)
def count_tokens_tiktoken(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def print_separator(title: str):
    print(f"\n{'='*75}")
    print(f" {title} ".center(75, '='))
    print(f"{'='*75}\n")

def main():
    print_separator("RAG BASICS: CHUNKING & RETRIEVAL")
    
    # Khởi tạo model trước để lấy tokenizer của model
    print("Đang tải model 'keepitreal/vietnamese-sbert'...")
    model = SentenceTransformer("keepitreal/vietnamese-sbert")
    tokenizer = model.tokenizer

    # Hàm đếm token theo đúng tokenizer của model
    def count_model_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    
    # Đọc file dữ liệu mẫu
    with open("sample_document.txt", "r", encoding="utf-8") as f:
        document_text = f.read()

    print(f"Tổng số ký tự: {len(document_text)}")
    print(f"Tổng số token (tiktoken - GPT): {count_tokens_tiktoken(document_text)}")
    print(f"Tổng số token (model thực tế) : {count_model_tokens(document_text)}")
    
    # 2. CÁC CHIẾN LƯỢC CHUNKING
    
    # Chiến lược 1: Fixed-size Character Chunking (Cắt cứng)
    fixed_splitter = CharacterTextSplitter(
        separator="", # Cắt không cần quan tâm dấu câu
        chunk_size=300,
        chunk_overlap=50
    )
    fixed_chunks = fixed_splitter.split_text(document_text)
    
    # Chiến lược 2: Recursive Character Chunking (Cắt đệ quy)
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=300,
        chunk_overlap=50
    )
    recursive_chunks = recursive_splitter.split_text(document_text)
    
    # Chiến lược 3: Structure-aware Markdown Chunking
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # GIỮ LẠI METADATA TRONG NỘI DUNG CHUNK
    )
    md_docs = markdown_splitter.split_text(document_text)
    
    # Tiếp tục chia nhỏ các phần bằng Recursive để đảm bảo kích thước không quá lớn
    structure_docs = recursive_splitter.split_documents(md_docs)
    structure_chunks = [doc.page_content for doc in structure_docs]

    # Hàm phụ để in thống kê độ dài
    def print_stats(name, chunks):
        lengths = [count_model_tokens(c) for c in chunks]
        print(f"{name:10} | {len(chunks):2} chunks | Token min/avg/max: {min(lengths):3} / {int(np.mean(lengths)):3} / {max(lengths):3}")

    print("\nSo sánh thống kê các chiến lược (đo bằng Token của model):")
    print_stats("Fixed", fixed_chunks)
    print_stats("Recursive", recursive_chunks)
    print_stats("Structure", structure_chunks)

    # 3. TẠO EMBEDDING VÀ SO SÁNH RETRIEVAL
    print_separator("TẠO EMBEDDING VÀ ĐÁNH GIÁ HIT@1")
    
    # Sinh embedding cho cả 3 chiến lược
    fixed_embs = model.encode(fixed_chunks, convert_to_numpy=True, normalize_embeddings=True)
    recursive_embs = model.encode(recursive_chunks, convert_to_numpy=True, normalize_embeddings=True)
    structure_embs = model.encode(structure_chunks, convert_to_numpy=True, normalize_embeddings=True)
    
    # 5 câu query thử nghiệm và nội dung kỳ vọng để chấm điểm Hit/Miss
    test_cases = [
        {
            "query": "Công ty làm việc vào sáng thứ mấy?",
            "expected_keyword": "Từ Thứ Hai đến Thứ Sáu"
        },
        {
            "query": "Chứng chỉ nào được đài thọ 100%?",
            "expected_keyword": "AWS Certified"
        },
        {
            "query": "Doanh thu dịch vụ AI tháng 1 là bao nhiêu?",
            "expected_keyword": "Doanh thu Dịch vụ AI"
        },
        {
            "query": "Lợi nhuận quý 1 đạt mức bao nhiêu?",
            "expected_keyword": "Tổng lợi nhuận trước thuế Quý 1 đạt"
        },
        {
            "query": "Có được dùng ChatGPT phân tích code không?",
            "expected_keyword": "ChatGPT, Claude"
        }
    ]
    
    # Hàm con để tìm chunk có điểm cao nhất và xác định Hit/Miss
    def evaluate_top_1(query_emb, doc_embs, chunks, expected_keyword):
        scores = doc_embs @ query_emb
        best_idx = np.argmax(scores)
        snippet = chunks[best_idx]
        is_hit = expected_keyword.lower() in snippet.lower()
        return is_hit
    
    # In Bảng kết quả (Format text table)
    print(f"{'Query':<45} | {'Fixed':<5} | {'Recurs':<6} | {'Struct':<6}")
    print("-" * 75)
    
    hit_counts = {"Fixed": 0, "Recurs": 0, "Struct": 0}
    
    for i, test in enumerate(test_cases):
        query = test["query"]
        expected = test["expected_keyword"]
        q_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        # Lấy đánh giá Top 1 của từng chiến lược
        f_hit = evaluate_top_1(q_emb, fixed_embs, fixed_chunks, expected)
        r_hit = evaluate_top_1(q_emb, recursive_embs, recursive_chunks, expected)
        s_hit = evaluate_top_1(q_emb, structure_embs, structure_chunks, expected)
        
        if f_hit: hit_counts["Fixed"] += 1
        if r_hit: hit_counts["Recurs"] += 1
        if s_hit: hit_counts["Struct"] += 1
        
        # Format string cho query (cắt ngắn nếu quá dài)
        q_str = f"Q{i+1}: {query}"
        if len(q_str) > 44: q_str = q_str[:41] + "..."
            
        print(f"{q_str:<45} | {'Hit' if f_hit else 'Miss':<5} | {'Hit' if r_hit else 'Miss':<6} | {'Hit' if s_hit else 'Miss':<6}")
        
    print("-" * 75)
    print(f"{'Tổng số câu đúng (Accuracy@1):':<45} | {hit_counts['Fixed']}/5   | {hit_counts['Recurs']}/5    | {hit_counts['Struct']}/5")
    
    print_separator("HOÀN TẤT DEMO")

if __name__ == "__main__":
    main()
