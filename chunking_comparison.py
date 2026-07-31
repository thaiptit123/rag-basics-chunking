import os
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

# 1. Đo độ dài token bằng tiktoken
def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def print_separator(title: str):
    print(f"\n{'='*55}")
    print(f" {title} ".center(55, '='))
    print(f"{'='*55}\n")

def main():
    print_separator("RAG BASICS: CHUNKING & RETRIEVAL")
    
    # Đọc file dữ liệu mẫu
    with open("sample_document.txt", "r", encoding="utf-8") as f:
        document_text = f.read()

    print(f"Tổng số ký tự: {len(document_text)}")
    print(f"Tổng số token (tiktoken): {count_tokens(document_text)}")
    
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
    
    # Chiến lược 3: Semantic/Structure-aware Chunking (Cắt theo cấu trúc Markdown)
    headers_to_split_on = [
        ("##", "Header 2")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = markdown_splitter.split_text(document_text)
    
    # Tái cấu trúc thành list string và lưu lại metadata để dễ in
    structure_chunks = []
    metadata_list = []
    for doc in md_docs:
        structure_chunks.append(doc.page_content)
        metadata_list.append(doc.metadata.get("Header 2", "Unknown"))

    print("\nSo sánh số lượng chunk tạo ra:")
    print(f"1. Fixed-size: {len(fixed_chunks)} chunks")
    print(f"2. Recursive : {len(recursive_chunks)} chunks")
    print(f"3. Structure : {len(structure_chunks)} chunks")

    # 3. TẠO EMBEDDING VÀ SO SÁNH RETRIEVAL
    print_separator("TẠO EMBEDDING VÀ RETRIEVAL")
    print("Đang tải model 'keepitreal/vietnamese-sbert'...")
    model = SentenceTransformer("keepitreal/vietnamese-sbert")
    
    # Sinh embedding cho cả 3 chiến lược
    fixed_embs = model.encode(fixed_chunks, convert_to_numpy=True, normalize_embeddings=True)
    recursive_embs = model.encode(recursive_chunks, convert_to_numpy=True, normalize_embeddings=True)
    structure_embs = model.encode(structure_chunks, convert_to_numpy=True, normalize_embeddings=True)
    
    # 5 câu query thử nghiệm
    queries = [
        "Công ty làm việc vào sáng thứ mấy?",
        "Chứng chỉ nào được đài thọ 100%?",
        "Doanh thu dịch vụ AI tháng 1 là bao nhiêu?",
        "Lợi nhuận quý 1 đạt mức bao nhiêu?",
        "Có được dùng ChatGPT phân tích code không?"
    ]
    
    # Hàm con để tìm chunk có điểm cao nhất
    def get_top_1(query_emb, doc_embs, chunks):
        scores = doc_embs @ query_emb
        best_idx = np.argmax(scores)
        snippet = chunks[best_idx].replace("\n", " ")
        snippet = snippet[:30] + "..." if len(snippet) > 30 else snippet
        return snippet, scores[best_idx]

    # Bảng kết quả (Format ngắn gọn dưới 55 ký tự để tránh tràn lề)
    print("KẾT QUẢ TRUY VẤN (TOP 1 CHUNK)")
    
    for i, query in enumerate(queries):
        print(f"\nQ{i+1}: {query}")
        q_emb = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        
        # Lấy Top 1 của từng chiến lược
        f_snip, f_score = get_top_1(q_emb, fixed_embs, fixed_chunks)
        r_snip, r_score = get_top_1(q_emb, recursive_embs, recursive_chunks)
        s_snip, s_score = get_top_1(q_emb, structure_embs, structure_chunks)
        
        print(f" Fixed : [{f_score:.2f}] {f_snip}")
        print(f" Recurs: [{r_score:.2f}] {r_snip}")
        print(f" Struct: [{s_score:.2f}] {s_snip}")
        
    print_separator("HOÀN TẤT DEMO")

if __name__ == "__main__":
    main()
